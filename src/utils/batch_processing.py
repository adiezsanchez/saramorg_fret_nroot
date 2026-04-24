import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tifffile
from tqdm.auto import tqdm

from utils.feature_extraction import (
    calculate_distance_to_root_surface,
    classify_root_cap_nuclei,
    compute_fret_ratios,
    extract_nuclei_depth,
    extract_nuclei_features_per_marker,
    map_root_body_depth_clusters_to_tissue_layers,
    merge_root_cap_into_tissue_layers,
)
from utils.inference import predict_tiled_unet
from utils.io import (
    calculate_rescale_factor,
    ensure_output_dir,
    explore_lif_container,
    get_voxel_spacing_zyx_um,
    list_containers,
    load_lif_image,
    load_precomputed_results_if_available,
)
from utils.segmentation import (
    fill_root_3d,
    generate_rough_root_3d,
    predict_nuclei_labels,
    simulate_fluo_from_bf,
    smooth_outer_root_surface_3d,
    wrap_outer_root_surface,
)


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip()
    return sanitized if sanitized else "unnamed"


def log_step(
    level: str,
    container_id: str,
    image_name: str,
    stage: str,
    msg: str,
    elapsed_s: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_text = "na" if elapsed_s is None else f"{elapsed_s:.2f}"
    details_text = "na" if details is None else json.dumps(details, default=str)

    line = (
        f"[{timestamp}] | {level} | container={container_id} | image={image_name} "
        f"| stage={stage} | msg={msg} | elapsed_s={elapsed_text} | details={details_text}"
    )
    tqdm.write(line)


def save_image_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _as_tuple_int(values: list[Any], field_name: str, expected_len: int) -> tuple[int, ...]:
    if not isinstance(values, list) or len(values) != expected_len:
        raise ValueError(f"'{field_name}' must be a list with length {expected_len}.")
    try:
        return tuple(int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain integer values.") from exc


def _as_optional_int_list(values: Any, field_name: str) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"'{field_name}' must be a list of integers or null.")
    try:
        return [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain integer values.") from exc


def _parse_markers(markers_raw: list[Any]) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(markers_raw, list) or len(markers_raw) == 0:
        raise ValueError("'markers' must be a non-empty list.")

    parsed_markers: list[tuple[str, int, str]] = []
    for marker in markers_raw:
        if not isinstance(marker, dict):
            raise ValueError("Each marker in 'markers' must be a mapping.")
        name = str(marker.get("name", "")).strip()
        role = str(marker.get("role", "")).strip()
        if not name or not role:
            raise ValueError("Each marker requires non-empty 'name' and 'role'.")
        try:
            channel = int(marker["channel"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each marker requires integer 'channel'.") from exc
        parsed_markers.append((name, channel, role))

    return tuple(parsed_markers)


def validate_runtime_config(config: dict[str, Any]) -> None:
    required_keys = {
        "raw_data_directory",
        "results_root",
        "model_dir",
        "markers",
        "nuclei_channel",
        "min_max_nuclei_volume",
        "root_probability_threshold",
        "root_occupancy_threshold",
        "root_fill_slice_aware",
        "root_smooth_erosion",
        "root_smoothing",
        "root_wrap_percentage_threshold",
        "root_wrap_edt_threshold",
        "depth_pad_full_root",
        "inference_patch",
        "inference_patch_halo",
        "inference_stride_ratio",
        "inference_batch_size",
        "inference_device",
        "inference_use_amp",
        "container_indices",
        "image_indices",
        "overwrite_csv",
    }
    missing = sorted(required_keys.difference(config.keys()))
    if missing:
        raise ValueError(f"Missing runtime config keys: {missing}")


def build_runtime_config(user_config: dict[str, Any], raw_data_directory: str | Path) -> dict[str, Any]:
    config = {
        "raw_data_directory": Path(raw_data_directory),
        "results_root": Path(user_config.get("results_root", "./results")),
        "model_dir": Path(user_config["model_dir"]),
        "markers": _parse_markers(user_config["markers"]),
        "nuclei_channel": int(user_config["nuclei_channel"]),
        "min_max_nuclei_volume": _as_tuple_int(
            user_config["min_max_nuclei_volume"], "min_max_nuclei_volume", expected_len=2
        ),
        "root_probability_threshold": float(user_config["root_probability_threshold"]),
        "root_occupancy_threshold": float(user_config["root_occupancy_threshold"]),
        "root_fill_slice_aware": bool(user_config["root_fill_slice_aware"]),
        "root_smooth_erosion": int(user_config["root_smooth_erosion"]),
        "root_smoothing": int(user_config["root_smoothing"]),
        "root_wrap_percentage_threshold": float(user_config["root_wrap_percentage_threshold"]),
        "root_wrap_edt_threshold": float(user_config["root_wrap_edt_threshold"]),
        "depth_pad_full_root": bool(user_config["depth_pad_full_root"]),
        "inference_patch": _as_tuple_int(user_config["inference_patch"], "inference_patch", expected_len=3),
        "inference_patch_halo": _as_tuple_int(
            user_config["inference_patch_halo"], "inference_patch_halo", expected_len=3
        ),
        "inference_stride_ratio": float(user_config["inference_stride_ratio"]),
        "inference_batch_size": int(user_config["inference_batch_size"]),
        "inference_device": str(user_config["inference_device"]),
        "inference_use_amp": bool(user_config["inference_use_amp"]),
        "container_indices": _as_optional_int_list(user_config.get("container_indices"), "container_indices"),
        "image_indices": _as_optional_int_list(user_config.get("image_indices"), "image_indices"),
        "overwrite_csv": bool(user_config.get("overwrite_csv", False)),
    }
    validate_runtime_config(config)
    return config


def process_single_image(
    lif_path: str,
    image_index: int,
    lif_container_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    image_start = time.perf_counter()
    image_name = f"image_{image_index}"

    try:
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "image_start",
            "starting image processing",
            elapsed_s=0.0,
        )

        stage_t0 = time.perf_counter()
        lif_image, lif_image_name, xml_metadata = load_lif_image(lif_path, image_index)
        image_name = lif_image_name
        safe_image_name = sanitize_filename(lif_image_name)
        csv_path = config["results_root"] / lif_container_id / f"{safe_image_name}.csv"

        elapsed = time.perf_counter() - image_start
        stage_elapsed = time.perf_counter() - stage_t0
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "load",
            "image and metadata loaded",
            elapsed_s=elapsed,
            details={"stage_s": round(stage_elapsed, 2), "image_index": image_index},
        )

        if csv_path.exists() and not config["overwrite_csv"]:
            elapsed = time.perf_counter() - image_start
            log_step(
                "WARN",
                lif_container_id,
                image_name,
                "skip",
                "csv exists and overwrite disabled",
                elapsed_s=elapsed,
                details={"csv": str(csv_path)},
            )
            return {
                "status": "skipped",
                "lif_image_name": lif_image_name,
                "csv_path": csv_path,
            }

        descriptor_dict = {
            "lif_container_id": lif_container_id,
            "lif_image_name": lif_image_name,
        }

        nuclei_labels_dir = ensure_output_dir(
            str(config["raw_data_directory"]),
            lif_container_id,
            results_type="nuclei_labels",
        )
        root_mask_dir = ensure_output_dir(
            str(config["raw_data_directory"]),
            lif_container_id,
            results_type="root_mask",
        )
        depth_map_dir = ensure_output_dir(
            str(config["raw_data_directory"]),
            lif_container_id,
            results_type="depth_map",
        )

        stage_t0 = time.perf_counter()
        nuclei_labels = load_precomputed_results_if_available(
            nuclei_labels_dir,
            safe_image_name,
            results_type="nuclei_labels",
        )

        if nuclei_labels is None:
            rescale_factor = calculate_rescale_factor(xml_metadata)
            nuclei_labels = predict_nuclei_labels(
                lif_image,
                rescale_factor,
                config["nuclei_channel"],
                config["min_max_nuclei_volume"],
                visualize=False,
            )
            nuclei_labels_path = nuclei_labels_dir / f"{safe_image_name}_nuclei_labels.tif"
            tifffile.imwrite(nuclei_labels_path, nuclei_labels)
            nuclei_source = "computed"
        else:
            nuclei_labels_path = nuclei_labels_dir / f"{safe_image_name}_nuclei_labels.tif"
            nuclei_source = "precomputed"

        elapsed = time.perf_counter() - image_start
        stage_elapsed = time.perf_counter() - stage_t0
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "nuclei",
            "nuclei labels ready",
            elapsed_s=elapsed,
            details={
                "stage_s": round(stage_elapsed, 2),
                "source": nuclei_source,
                "n_labels": int(nuclei_labels.max()),
                "nuclei_labels_tif": str(nuclei_labels_path),
            },
        )

        stage_t0 = time.perf_counter()
        props_df = extract_nuclei_features_per_marker(
            nuclei_labels,
            lif_image,
            config["markers"],
            descriptor_dict,
        )
        props_df = compute_fret_ratios(props_df, config["markers"])
        elapsed = time.perf_counter() - image_start
        stage_elapsed = time.perf_counter() - stage_t0
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "features",
            "feature extraction and FRET computation complete",
            elapsed_s=elapsed,
            details={"stage_s": round(stage_elapsed, 2), "n_rows": int(len(props_df))},
        )

        stage_t0 = time.perf_counter()
        root_body_3d_mask = load_precomputed_results_if_available(
            root_mask_dir,
            safe_image_name,
            results_type="root_mask",
        )

        if root_body_3d_mask is None:
            sim_fluo_cell_walls = simulate_fluo_from_bf(lif_image, config["markers"])
            root_pmaps = predict_tiled_unet(
                raw=sim_fluo_cell_walls,
                input_layout="ZYX",
                model_dir=config["model_dir"],
                patch=config["inference_patch"],
                patch_halo=config["inference_patch_halo"],
                stride_ratio=config["inference_stride_ratio"],
                batch_size=config["inference_batch_size"],
                device=config["inference_device"],
                use_amp=config["inference_use_amp"],
            )
            rough_root_3d = generate_rough_root_3d(
                root_pmaps,
                nuclei_labels,
                probability_threshold=config["root_probability_threshold"],
                visualize=False,
                remove_nonconnected_components=False,
            )
            filled_root_3d = fill_root_3d(
                rough_root_3d,
                occupancy_threshold=config["root_occupancy_threshold"],
                slice_aware_filling=config["root_fill_slice_aware"],
                visualize=False,
            )
            smooth_root_3d = smooth_outer_root_surface_3d(
                filled_root_3d,
                erosion=config["root_smooth_erosion"],
                smoothing=config["root_smoothing"],
                visualize=False,
            )
            props_df = classify_root_cap_nuclei(props_df)
            root_body_3d_mask = wrap_outer_root_surface(
                nuclei_labels,
                smooth_root_3d,
                props_df,
                percentage_threshold=config["root_wrap_percentage_threshold"],
                edt_threshold=config["root_wrap_edt_threshold"],
                visualize=False,
            )
            root_mask_path = root_mask_dir / f"{safe_image_name}_root_mask.tif"
            tifffile.imwrite(root_mask_path, root_body_3d_mask)
            root_mask_source = "computed"
        else:
            props_df = classify_root_cap_nuclei(props_df)
            root_mask_path = root_mask_dir / f"{safe_image_name}_root_mask.tif"
            root_mask_source = "precomputed"

        elapsed = time.perf_counter() - image_start
        stage_elapsed = time.perf_counter() - stage_t0
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "root_mask",
            "root mask and root-part assignment complete",
            elapsed_s=elapsed,
            details={
                "stage_s": round(stage_elapsed, 2),
                "source": root_mask_source,
                "root_mask_tif": str(root_mask_path),
            },
        )

        stage_t0 = time.perf_counter()
        nuclei_depth_map = load_precomputed_results_if_available(
            depth_map_dir,
            safe_image_name,
            results_type="depth_map",
        )

        if nuclei_depth_map is None:
            spacing_zyx_um = get_voxel_spacing_zyx_um(xml_metadata)
            nuclei_depth_map, is_flooded, flooded_planes = calculate_distance_to_root_surface(
                nuclei_labels,
                root_body_3d_mask,
                pad_full_root=config["depth_pad_full_root"],
                spacing_zyx_um=spacing_zyx_um,
                visualize=False,
            )
            depth_map_path = depth_map_dir / f"{safe_image_name}_depth_map.tif"
            tifffile.imwrite(depth_map_path, nuclei_depth_map)
            depth_map_source = "computed"
        else:
            is_flooded = False
            flooded_planes = []
            depth_map_path = depth_map_dir / f"{safe_image_name}_depth_map.tif"
            depth_map_source = "precomputed"

        depth_df = extract_nuclei_depth(nuclei_labels, nuclei_depth_map)
        props_df = props_df.merge(depth_df, on="label")

        depth_clusters_df = map_root_body_depth_clusters_to_tissue_layers(props_df)
        props_df = merge_root_cap_into_tissue_layers(props_df, depth_clusters_df)
        elapsed = time.perf_counter() - image_start
        stage_elapsed = time.perf_counter() - stage_t0
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "depth",
            "depth map and tissue layers computed",
            elapsed_s=elapsed,
            details={
                "stage_s": round(stage_elapsed, 2),
                "source": depth_map_source,
                "is_flooded": bool(is_flooded),
                "flooded_planes": int(len(flooded_planes)),
                "depth_map_tif": str(depth_map_path),
            },
        )

        stage_t0 = time.perf_counter()
        save_image_csv(props_df, csv_path)
        elapsed = time.perf_counter() - image_start
        stage_elapsed = time.perf_counter() - stage_t0
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "write_csv",
            "csv written",
            elapsed_s=elapsed,
            details={"stage_s": round(stage_elapsed, 2), "csv": str(csv_path)},
        )

        total_elapsed = time.perf_counter() - image_start
        log_step(
            "INFO",
            lif_container_id,
            image_name,
            "done",
            "image processing completed",
            elapsed_s=total_elapsed,
            details={"n_rows": int(len(props_df))},
        )

        return {
            "status": "success",
            "lif_image_name": lif_image_name,
            "csv_path": csv_path,
            "n_rows": int(len(props_df)),
        }

    except Exception as exc:
        total_elapsed = time.perf_counter() - image_start
        log_step(
            "ERROR",
            lif_container_id,
            image_name,
            "failed",
            "processing error",
            elapsed_s=total_elapsed,
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        tqdm.write(traceback.format_exc())
        return {
            "status": "failed",
            "lif_image_name": image_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_batch(config: dict[str, Any]) -> dict[str, Any]:
    validate_runtime_config(config)
    batch_start = time.perf_counter()
    config["results_root"].mkdir(parents=True, exist_ok=True)

    lif_containers = list_containers(str(config["raw_data_directory"]), file_format="lif")
    if not lif_containers:
        raise FileNotFoundError(
            f"No .lif containers found in '{config['raw_data_directory']}'."
        )

    if config["container_indices"] is None:
        selected_containers = lif_containers
    else:
        selected_containers = [lif_containers[i] for i in config["container_indices"]]

    processed = 0
    success = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []

    container_pbar = tqdm(selected_containers, desc="Containers", unit="container")
    for lif_path in container_pbar:
        nr_imgs, lif_container_id = explore_lif_container(lif_path, display=False)

        if config["image_indices"] is None:
            image_indices = list(range(nr_imgs))
        else:
            image_indices = [i for i in config["image_indices"] if i < nr_imgs]

        image_pbar = tqdm(
            image_indices,
            desc=f"Images[{lif_container_id}]",
            unit="image",
            leave=False,
        )

        for image_index in image_pbar:
            result = process_single_image(
                lif_path=lif_path,
                image_index=image_index,
                lif_container_id=lif_container_id,
                config=config,
            )

            processed += 1
            status = result["status"]
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(
                    {
                        "container": lif_container_id,
                        "image": str(result.get("lif_image_name", f"image_{image_index}")),
                        "error_type": str(result.get("error_type", "UnknownError")),
                        "error": str(result.get("error", "Unknown error")),
                    }
                )

            image_pbar.set_postfix(
                processed=processed,
                success=success,
                skipped=skipped,
                failed=failed,
            )
            container_pbar.set_postfix(
                container=lif_container_id,
                processed=processed,
                success=success,
                skipped=skipped,
                failed=failed,
            )

    elapsed_min = (time.perf_counter() - batch_start) / 60.0
    tqdm.write(
        "BATCH_SUMMARY"
        f" | containers={len(selected_containers)}"
        f" | images_total={processed}"
        f" | processed={processed}"
        f" | success={success}"
        f" | skipped={skipped}"
        f" | failed={failed}"
        f" | elapsed_min={elapsed_min:.2f}"
    )

    if failures:
        for item in failures:
            tqdm.write(
                "FAILED_ITEM"
                f" | container={item['container']}"
                f" | image={item['image']}"
                f" | error_type={item['error_type']}"
                f" | error={item['error']}"
            )

    return {
        "containers": len(selected_containers),
        "images_total": processed,
        "processed": processed,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "elapsed_min": elapsed_min,
        "failures": failures,
    }
