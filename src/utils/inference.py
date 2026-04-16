from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import yaml
from utils.model import UNet2D, UNet3D


MODEL_REGISTRY = {
    "UNet3D": UNet3D,
    "UNet2D": UNet2D,
}


def _remap_state_dict_for_model(
    state: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """
    Handle minor checkpoint naming drifts across PanSeg/pytorch3dunet variants.

    Example seen in the wild:
      - checkpoint keys: ...basic_module.SingleConv1...
      - model keys:      ...basic_module.single_conv1...
    """
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    remap_pairs = (
        ("SingleConv1", "single_conv1"),
        ("SingleConv2", "single_conv2"),
    )

    def _apply_pairs(keys: set[str], pairs: tuple[tuple[str, str], ...]) -> set[str]:
        out = set()
        for k in keys:
            kk = k
            for src, dst in pairs:
                kk = kk.replace(src, dst)
            out.add(kk)
        return out

    baseline_overlap = len(model_keys.intersection(state_keys))
    remapped_overlap = len(model_keys.intersection(_apply_pairs(state_keys, remap_pairs)))

    if remapped_overlap > baseline_overlap:
        remapped_state = {}
        for k, v in state.items():
            kk = k
            for src, dst in remap_pairs:
                kk = kk.replace(src, dst)
            remapped_state[kk] = v
        return remapped_state

    return state


def fix_layout_to_zyx(data: np.ndarray, input_layout: str) -> np.ndarray:
    if input_layout == "ZYX":
        out = data
    elif input_layout == "YX":
        out = data[None, ...]
    elif input_layout == "CZYX":
        if data.shape[0] != 1:
            raise ValueError("CZYX -> ZYX requires single input channel")
        out = data[0]
    elif input_layout == "CYX":
        if data.shape[0] != 1:
            raise ValueError("CYX -> ZYX requires single input channel")
        out = data
    else:
        raise ValueError(f"Unsupported input_layout: {input_layout}")
    if out.ndim != 3:
        raise ValueError(f"Expected ZYX, got {out.shape}")
    return out


def fix_layout_to_czyx(data: np.ndarray, input_layout: str) -> np.ndarray:
    if input_layout == "ZYX":
        out = data[None, ...]
    elif input_layout == "YX":
        out = data[None, None, ...]
    elif input_layout == "CZYX":
        out = data
    elif input_layout == "CYX":
        out = data[:, None, ...]
    else:
        raise ValueError(f"Unsupported input_layout: {input_layout}")
    if out.ndim != 4:
        raise ValueError(f"Expected CZYX, got {out.shape}")
    return out


def zscore_global(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return (x - x.mean()) / (x.std() + eps)


def _gen_indices(i: int, k: int, s: int) -> Iterable[int]:
    if i < k:
        raise ValueError(f"Patch size {k} > dimension {i}")
    j = 0
    while j <= i - k:
        yield j
        j += s
    if (j - s) + k < i:
        yield i - k


def _build_spatial_slices(
    shape_zyx: tuple[int, int, int],
    patch_zyx: tuple[int, int, int],
    stride_zyx: tuple[int, int, int],
):
    iz, iy, ix = shape_zyx
    kz, ky, kx = patch_zyx
    sz, sy, sx = stride_zyx
    out = []
    for z in _gen_indices(iz, kz, sz):
        for y in _gen_indices(iy, ky, sy):
            for x in _gen_indices(ix, kx, sx):
                out.append((slice(z, z + kz), slice(y, y + ky), slice(x, x + kx)))
    return out


def _mirror_pad_czyx(raw_czyx: np.ndarray, halo_zyx: tuple[int, int, int]) -> np.ndarray:
    if all(h == 0 for h in halo_zyx):
        return raw_czyx
    return np.pad(
        raw_czyx,
        [
            (0, 0),
            (halo_zyx[0], halo_zyx[0]),
            (halo_zyx[1], halo_zyx[1]),
            (halo_zyx[2], halo_zyx[2]),
        ],
        mode="reflect",
    )


def _remove_halo(pred_bczyx: torch.Tensor, halo_zyx: tuple[int, int, int]) -> torch.Tensor:
    hz, hy, hx = halo_zyx
    z_slice = slice(hz, -hz if hz > 0 else None)
    y_slice = slice(hy, -hy if hy > 0 else None)
    x_slice = slice(hx, -hx if hx > 0 else None)
    return pred_bczyx[:, :, z_slice, y_slice, x_slice]


def load_model_from_folder(model_dir: str | Path, device: str = "cuda"):
    model_dir = Path(model_dir)
    config_path = model_dir / "config_train.yml"
    weights_path = model_dir / "best_checkpoint.pytorch"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model class in config: {model_name}")

    model = MODEL_REGISTRY[model_name](**model_cfg)

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = _remap_state_dict_for_model(state, model)
    model.load_state_dict(state)

    model.eval().to(device)
    return model, cfg


@torch.no_grad()
def predict_tiled_unet(
    raw: np.ndarray,
    input_layout: str,
    model_dir: str | Path,
    patch: tuple[int, int, int] = (80, 160, 160),
    patch_halo: tuple[int, int, int] = (0, 0, 0),
    stride_ratio: float = 0.75,
    batch_size: int = 1,
    device: str = "cuda",
    use_amp: bool = True,
) -> np.ndarray:
    """
    PanSeg-like inference:
      - global z-score
      - tiled patch inference
      - optional halo context per tile
      - overlap averaging blend

    Returns:
      np.ndarray shape: (C_out, Z, Y, X)
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    model, cfg = load_model_from_folder(model_dir, device=device)
    model_name = cfg["model"]["name"]
    in_channels = int(cfg["model"]["in_channels"])
    out_channels = int(cfg["model"]["out_channels"])
    is_2d = model_name == "UNet2D"

    if is_2d and patch[0] != 1:
        patch = (1, patch[1], patch[2])

    if in_channels > 1:
        raw_czyx = fix_layout_to_czyx(raw, input_layout)
    else:
        raw_zyx = fix_layout_to_zyx(raw, input_layout)
        raw_czyx = raw_zyx[None, ...]

    raw_czyx = zscore_global(raw_czyx)
    volume_shape = raw_czyx.shape[1:]

    stride = tuple(max(int(p * stride_ratio), 1) for p in patch)
    slices_zyx = _build_spatial_slices(volume_shape, patch, stride)

    raw_padded = _mirror_pad_czyx(raw_czyx, patch_halo)

    pred_sum = np.zeros((out_channels,) + volume_shape, dtype=np.float32)
    pred_count = np.zeros((out_channels,) + volume_shape, dtype=np.float32)

    def _extract_patch_with_halo(slc):
        z, y, x = slc
        hz, hy, hx = patch_halo
        return raw_padded[
            :,
            z.start : z.stop + 2 * hz,
            y.start : y.stop + 2 * hy,
            x.start : x.stop + 2 * hx,
        ]

    for i in range(0, len(slices_zyx), batch_size):
        batch_slices = slices_zyx[i : i + batch_size]
        batch_np = np.stack(
            [_extract_patch_with_halo(s) for s in batch_slices], axis=0
        )
        x = torch.from_numpy(batch_np).to(device, non_blocking=True)

        if is_2d:
            b, c, z, yy, xx = x.shape
            x2d = x.permute(0, 2, 1, 3, 4).reshape(b * z, c, yy, xx)
            with torch.autocast(
                device_type="cuda", enabled=(use_amp and device.startswith("cuda"))
            ):
                y2d = model(x2d)
            cout = y2d.shape[1]
            y = y2d.reshape(b, z, cout, yy, xx).permute(0, 2, 1, 3, 4)
        else:
            with torch.autocast(
                device_type="cuda", enabled=(use_amp and device.startswith("cuda"))
            ):
                y = model(x)

        y = _remove_halo(y, patch_halo)
        y_np = y.float().cpu().numpy()

        for tile_pred, slc in zip(y_np, batch_slices):
            z, y_, x_ = slc
            idx = (slice(None), z, y_, x_)
            pred_sum[idx] += tile_pred
            pred_count[idx] += 1.0

    return pred_sum / np.maximum(pred_count, 1e-8)
