import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def plot_prop_to_3d_centroids(
    props_df,
    value_column="depth_cluster_id",
    colormap="turbo",
    nuclei_labels_shape=None,
    fig_title=None,
    ax_labels=None,
    colormap_vmin=None,
    colormap_vmax=None,
    save_fig=False,
    fig_filename=None,
    fig_savepath=None,
    visualize=False
):
    """
    Visualizes per-nucleus property by mapping values from a DataFrame to 3D centroid scatter plot.
    Similar in logic to map_df_column_to_labels but uses 3D centroid scatter, not Napari nor nuclei_labels.

    Args:
        props_df (pd.DataFrame): DataFrame with centroid columns and a value column for coloring.
        value_column (str): column name to map as color. E.g. "depth_cluster_id", "FRET_ratio", etc.
        colormap (str): valid matplotlib colormap name. E.g. "turbo" or "viridis".
        nuclei_labels_shape (tuple or None): The (z, y, x) shape used for axis scaling. If None, scales auto.
        fig_title (str or None): Custom title. If None, uses value_column.
        ax_labels (tuple or None): Optional custom axis labels as (xlabel, ylabel, zlabel).
        colormap_vmin (float or None): Colormap min. If None, uses min(props_df[value_column]).
        colormap_vmax (float or None): Colormap max. If None, uses max(props_df[value_column]).
        save_fig (bool): If True, save the plot as a PNG using fig_filename and fig_savepath.
        fig_filename (str or None): Name of the file to save (should end with .png). Required if save_fig is True.
        fig_savepath (str or None): Directory to save figure. Required if save_fig is True.
    """
    # Extract centroid columns (assumes columns named centroid-2,1,0 for X,Y,Z respectively)
    x = props_df["centroid-2"]  # X spatial axis (axis=2)
    y = props_df["centroid-1"]  # Y spatial axis (axis=1)
    z = props_df["centroid-0"]  # Z spatial axis (axis=0)
    values = props_df[value_column]

    vmin = colormap_vmin if colormap_vmin is not None else np.nanmin(values)
    vmax = colormap_vmax if colormap_vmax is not None else np.nanmax(values)
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(values))

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits if shape provided
    if nuclei_labels_shape is not None and len(nuclei_labels_shape) == 3:
        # (z, y, x) order
        zlim, ylim, xlim = nuclei_labels_shape
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.set_zlim(0, zlim)
        try:
            ax.set_box_aspect([xlim, ylim, zlim])
        except Exception:
            max_dim = max([xlim, ylim, zlim])
            ax.set_xlim(0, xlim * max_dim / xlim)
            ax.set_ylim(0, ylim * max_dim / ylim)
            ax.set_zlim(0, zlim * max_dim / zlim)

    scatter = ax.scatter(x, y, z, c=colors, alpha=0.6)

    # Axis labels (optionally override)
    if ax_labels and len(ax_labels) == 3:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_zlabel(ax_labels[2])
    else:
        ax.set_xlabel("centroid-2 (X)")
        ax.set_ylabel("centroid-1 (Y)")
        ax.set_zlabel("centroid-0 (Z)")

    # Title
    if fig_title is not None:
        ax.set_title(fig_title)
    else:
        ax.set_title(f"3D scatter colored by {value_column}")

    # Add colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.4)
    cbar.set_label(value_column)
    # If coloring by integer clusters (like depth_cluster_id), set ticks as integers
    if np.issubdtype(values.dtype, np.integer) and (vmin > 0) and (vmax < 20):
        cbar.set_ticks(np.arange(int(np.floor(vmin)), int(np.ceil(vmax))+1))
        cbar.set_ticklabels([str(i) for i in range(int(np.floor(vmin)), int(np.ceil(vmax))+1)])

    if save_fig:
        # Check for required arguments
        if fig_filename is None or fig_savepath is None:
            raise ValueError("fig_filename and fig_savepath must be specified when save_fig=True")
        import os
        figpath = os.path.join(fig_savepath, fig_filename)
        fig.savefig(figpath, bbox_inches='tight', dpi=200)
        print(f"Figure saved to: {figpath}")

    if visualize:
        fig.show()