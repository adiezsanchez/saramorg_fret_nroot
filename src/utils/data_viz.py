import numpy as np
import pandas as pd
from typing import Optional, Tuple

def _resolve_napari_viewer(viewer):
    """
    Return a napari Viewer for optional visualization.

    If ``viewer`` is passed, it is used. Otherwise tries ``napari.current_viewer()``;
    if none exists, creates ``napari.Viewer()``. Import is deferred until visualization runs.
    """
    if viewer is not None:
        return viewer
    import napari

    v = napari.current_viewer()
    if v is not None:
        return v
    return napari.Viewer()

def map_df_column_to_labels(
    nuclei_labels: np.ndarray,
    df: "pd.DataFrame",
    value_column: str,
    label_column: str = "label",
    normalize: bool = False,
    clip_percentiles: Optional[Tuple[float, float]] = (1, 99),
    background_value: float = 0.0,
    colormap: str = "turbo",   
    visualize: bool = False,
    viewer=None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Efficiently map per-label values from a DataFrame to a labeled image using vectorized lookup.

    This function assigns a value (e.g., FRET ratio) to each voxel in a labeled image
    (`nuclei_labels`) based on a lookup table constructed from a pandas DataFrame.
    It avoids per-label masking loops by using a NumPy array for direct indexing,
    making it suitable for large 2D/3D images.

    Args:
        nuclei_labels (np.ndarray):
            Labeled image where each integer value corresponds to a nucleus (0 = background).

        df (pd.DataFrame):
            DataFrame containing per-label measurements. Must include `label_column`
            and `value_column`.

        value_column (str):
            Column in `df` containing values to map (e.g., 'FRET_ratio_sum_norm').

        label_column (str, optional):
            Column in `df` containing label IDs. Default is "label".

        normalize (bool, optional):
            If True, normalize mapped values to [0, 1] after optional percentile clipping.
            Use this for raw ratios. If values are already normalized, keep False.

        clip_percentiles (tuple or None, optional):
            Percentiles (low, high) for clipping before normalization (e.g., (1, 99)).
            Set to None to disable clipping.

        background_value (float, optional):
            Value assigned to background (label 0). Default is 0.0.

        colormap (str, optional):
            Colormap to use for visualization. Default is "turbo".
            
        visualize (bool, optional):
            If True, display the result in Napari.

        viewer (napari.Viewer, optional):
            Existing Napari viewer. If None, a new one will be created.

        layer_name (str, optional):
            Name of the Napari layer. Defaults to `value_column`.

    Returns:
        np.ndarray:
            Image of same shape as `nuclei_labels`, where each voxel contains
            the mapped value corresponding to its label.
    """

    # --- Build lookup table (label -> value) ---
    max_label = int(nuclei_labels.max())
    lookup = np.full(max_label + 1, background_value, dtype=float)

    # Fill lookup table using dataframe values
    for _, row in df.iterrows():
        label_id = int(row[label_column])
        if label_id <= max_label:
            value = row[value_column]
            if not np.isnan(value):
                lookup[label_id] = value

    # --- Vectorized mapping ---
    # Each voxel gets value = lookup[label]
    out = lookup[nuclei_labels]

    # --- Optional normalization ---
    if normalize:
        mask = nuclei_labels > 0

        if np.any(mask):
            values = out[mask]

            # Optional percentile clipping
            if clip_percentiles is not None:
                p_low, p_high = np.percentile(values, clip_percentiles)
                values = np.clip(values, p_low, p_high)

            # Normalize to [0, 1]
            min_val = values.min()
            max_val = values.max()

            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.zeros_like(values)

            out[mask] = values

    # --- Optional visualization in Napari ---
    if visualize:
        v = _resolve_napari_viewer(viewer)
        name = layer_name if layer_name is not None else value_column
        v.add_image(
            out,
            name=name,
            colormap=colormap,
            blending="additive",
        )

    return out