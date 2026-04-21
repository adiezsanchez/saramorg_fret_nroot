import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, regionprops_table


MORPHOLOGY_PROPERTIES = [
    "label",
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "solidity",
    "inertia_tensor_eigvals",
]

INTENSITY_PROPERTIES = [
    "label",
    "intensity_mean",
    "intensity_min",
    "intensity_max",
    "intensity_std",
]

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

def _flood_fill_planes_below_threshold(
    mask_3d: np.ndarray, ratio_threshold: float = 0.7
) -> tuple[np.ndarray, bool, list[int]]:
    """
    Fill weakly populated planes in the final 15% of the stack.

    For planes in the last 15% of Z, if
    ``(true_pixels_plane / max_true_pixels_all_planes) < ratio_threshold``,
    replace that plane with the shape from the most filled plane.

    Args:
        mask_3d (np.ndarray): 3D boolean array.
        ratio_threshold (float, optional): Fill threshold ratio. Defaults to 0.7.

    Returns:
        tuple[np.ndarray, bool, list[int]]:
            - Updated mask
            - ``is_flooded`` (True if any plane was filled)
            - ``filled_planes`` (indices of planes that were filled)
    """
    mask_3d_filled = mask_3d.copy()
    n_planes = mask_3d.shape[0]
    true_counts = np.count_nonzero(mask_3d, axis=(1, 2))
    max_true_pixels = true_counts.max()

    if max_true_pixels == 0:
        return mask_3d_filled, False, []

    ratios = true_counts / max_true_pixels
    n_last_planes = max(1, int(np.ceil(n_planes * 0.15)))
    last_planes = np.arange(n_planes - n_last_planes, n_planes)
    failing_planes = [plane for plane in last_planes if ratios[plane] < ratio_threshold]

    if not failing_planes:
        return mask_3d_filled, False, []

    best_plane_idx = int(np.argmax(true_counts))
    best_plane_shape = mask_3d[best_plane_idx]

    for plane_idx in failing_planes:
        mask_3d_filled[plane_idx] = best_plane_shape

    return mask_3d_filled, True, failing_planes

def _pad_half_root(root_3d_mask: np.ndarray) -> np.ndarray:
    """
    Pads the root 3D mask with a border of zeros everywhere EXCEPT on the last Z slice (middle of the root).
    Pads Z only on the lower end (start), not on the upper end (stop); pads Y and X on both sides.
    Computes the distance transform on the padded mask and returns the distance map with padding removed.

    Args
        root_3d_mask (np.ndarray) Boolean 3D mask (Z, Y, X) of the root body.

    Returns
        np.ndarray Distance map with the same shape as the original mask, after padding and unpadding.
    """
    pad_width = ((1, 0), (1, 1), (1, 1))
    mask_padded = np.pad(
        root_3d_mask.astype(bool),
        pad_width=pad_width,
        mode='constant',
        constant_values=0
    )
    dist_padded = distance_transform_edt(mask_padded)
    dist_map = dist_padded[
        1 : 1 + root_3d_mask.shape[0],
        1 : 1 + root_3d_mask.shape[1],
        1 : 1 + root_3d_mask.shape[2]
    ]
    return dist_map

def _pad_full_root(root_3d_mask: np.ndarray) -> np.ndarray:
    """
    Pad the entire root 3D mask with a border of zeros on all sides.

    Useful for cases where the entire root (or root cone) is being analyzed,
    ensuring boundary proximity is detected at all edges of the root.
    The mask is padded by one voxel in all three dimensions.
    Computes the distance transform on the padded mask and removes the padding to return a distance map of the original shape.

    Args:
        root_3d_mask (np.ndarray): Boolean 3D mask (Z, Y, X) representing the root body.

    Returns:
        np.ndarray: Distance map with the same shape as the original mask, after full padding and unpadding.
    """
    mask_padded = np.pad(root_3d_mask.astype(bool), pad_width=1, mode='constant', constant_values=0)
    dist_padded = distance_transform_edt(mask_padded)

    # Remove padding to restore the shape to original spatial dimensions.
    dist_map = dist_padded[1:-1, 1:-1, 1:-1]

    return dist_map

def calculate_distance_to_root_surface(
    nuclei_labels: np.ndarray,
    root_3d_mask: np.ndarray,
    pad_full_root: bool = False,
    visualize: bool = False,
    viewer=None
) -> tuple[np.ndarray, bool, list[int]]:
    """
    Calculate a normalized distance map from the root surface and assign a depth value to each nucleus label, based on its centroid.

    Depending on the value of `pad_full_root`, will pad either the entire root mask (full root cone) or just the lower/side borders (half root cone).
    The normalized distance is defined as the distance from each point inside the root mask to the nearest root surface, divided by the maximum thickness.

    Optionally display the result in the global Napari viewer.

    Args:
        nuclei_labels (np.ndarray): 3D label array of nuclei (Z, Y, X).
        root_3d_mask (np.ndarray): Boolean 3D mask (Z, Y, X) representing the root body.
        pad_full_root (bool, optional): If True, pad the entire root mask (full root cone). If False, pad only the lower Z and the sides (half root cone). Defaults to False.
        visualize (bool, optional): If True, display the normalized depth map in Napari.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.

    Returns:
        tuple[np.ndarray, bool, list[int]]:
            - 3D array with normalized per-nucleus depth values (zero outside labels)
            - ``is_flooded`` indicating if tail planes were flood-filled
            - ``flooded_planes`` containing indices of flood-filled planes
    """
    # Choose padding strategy depending on root type (full or half cone)
    if pad_full_root:
        dist_map = _pad_full_root(root_3d_mask)
        is_flooded = False
        flooded_planes = []
    else:
        # Compensate for truncated tails before half-root padding.
        root_3d_mask, is_flooded, flooded_planes = _flood_fill_planes_below_threshold(
            root_3d_mask
        )
        dist_map = _pad_half_root(root_3d_mask)

    # Normalize distances by the maximum value inside the root (approximate root radius).
    max_dist = dist_map.max()
    if max_dist == 0:
        raise ValueError("Distance map is empty. Check mask.")

    dist_norm = dist_map / max_dist

    # Compute depth for each nucleus label based on the centroid coordinates in the normalized distance map.
    depth_per_label = {}
    props = regionprops(nuclei_labels)
    for prop in props:
        label_id = prop.label
        z, y, x = map(int, prop.centroid)  # Centroid coordinates
        depth = dist_norm[z, y, x]
        depth_per_label[label_id] = depth

    # Create an image where each voxel in a nucleus gets the assigned depth value for its label.
    depth_image = np.zeros_like(dist_map, dtype=float)
    for label_id, depth in depth_per_label.items():
        depth_image[nuclei_labels == label_id] = depth

    # Enhance contrast (clip the depth values between 1st and 99th percentile and re-normalize).
    nonzero = depth_image > 0
    if np.any(nonzero):
        p1, p99 = np.percentile(depth_image[nonzero], (1, 99))
        depth_image = np.clip(depth_image, p1, p99)
    depth_image = (depth_image - depth_image.min()) / (
        depth_image.max() - depth_image.min() + 1e-8
    )

    # Display the resulting normalized per-nucleus depth in Napari if requested.
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_image(depth_image, name="nuclei_depth_normalized", colormap="viridis", blending="additive")

    return depth_image, is_flooded, flooded_planes

def extract_nuclei_features_per_marker(
    nuclei_labels: np.ndarray,
    lif_image: np.ndarray,
    markers: list[tuple[str, int]],
    descriptor_dict: dict[str, str | int | float | bool],
) -> pd.DataFrame:
    """
    Extract morphology and per-marker intensity features for each nucleus label.

    The function computes a base morphology table from ``nuclei_labels`` using
    ``MORPHOLOGY_PROPERTIES`` and then appends per-channel intensity statistics for
    all markers except ``"brightfield"`` using ``INTENSITY_PROPERTIES``.
    Descriptor metadata are inserted as leading columns in the returned dataframe.

    Args:
        nuclei_labels (np.ndarray): 3D label image where each nucleus has a unique integer label.
        lif_image (np.ndarray): Multichannel image array indexed as ``lif_image[channel]``.
        markers (list[tuple[str, int]]): Marker definitions as ``(marker_name, channel_index)``.
        descriptor_dict (dict[str, str | int | float | bool]): Metadata values to prepend as columns.

    Returns:
        pd.DataFrame: Per-nucleus feature table containing morphology, per-marker
            intensity features, and descriptor metadata.
    """
    # Compute base morphology properties table for each nucleus
    props_morphology = regionprops_table(
        label_image=nuclei_labels,
        properties=MORPHOLOGY_PROPERTIES,
    )
    # Convert the properties dictionary to a DataFrame
    props_df = pd.DataFrame(props_morphology)

    # Iterate over all markers to extract intensity features
    for marker_name, ch_nr in markers:
        # Skip the brightfield marker (no intensity features required)
        if marker_name == "brightfield":
            continue

        # Compute intensity features for this marker channel
        props = regionprops_table(
            label_image=nuclei_labels,
            intensity_image=lif_image[ch_nr],  # The marker's image channel
            properties=INTENSITY_PROPERTIES,
        )
        intensity_df = pd.DataFrame(props)

        # Construct a renaming map for the intensity columns to include the marker's name
        rename_map = {"label": "label"}
        for prop in INTENSITY_PROPERTIES:
            if prop == "label":
                continue
            # For each intensity property, add the marker_name as a prefix
            if prop.startswith("intensity_"):
                suffix = prop.replace("intensity_", "")
                rename_map[prop] = f"{marker_name}_{suffix}_int"

        # Rename columns in the intensity DataFrame
        intensity_df.rename(columns=rename_map, inplace=True)
        # Merge the current marker's intensity features into the main DataFrame
        props_df = props_df.merge(intensity_df, on="label")

        # Derived columns section (per-marker columns like markerX_sum_int)
        mean_col = rename_map["intensity_mean"]
        area_col = "area"
        # Calculate total marker content per cell (mean_intensity * area)
        props_df[f"{marker_name}_sum_int"] = props_df[mean_col] * props_df[area_col]

    # Insert metadata columns at the beginning of the DataFrame, preserving input order
    insertion_position = 0
    for key, value in descriptor_dict.items():
        props_df.insert(insertion_position, key, value)
        insertion_position += 1

    # Return the final DataFrame with morphology, marker intensity features, and metadata
    return props_df

def extract_nuclei_depth(
    nuclei_labels: np.ndarray,
    nuclei_depth_map: np.ndarray
) -> pd.DataFrame:
    """
    Extracts the depth value of each nucleus from the provided depth map.

    For each nucleus labeled in the `nuclei_labels` array, this function computes the mean intensity 
    (depth) from the corresponding region in the `nuclei_depth_map` and returns a DataFrame 
    containing label and depth columns.

    Args:
        nuclei_labels (np.ndarray): 3D array with unique integer labels for each nucleus.
        nuclei_depth_map (np.ndarray): 3D array of the same shape as `nuclei_labels`,
            containing the per-voxel depth value (0-1, normalized).

    Returns:
        pd.DataFrame: DataFrame with columns "label" and "depth", one row per nucleus.
    """
    # Calculate the mean depth for each label (nucleus).
    # All pixels in each nucleus labels have the same depth value, so we can use the mean intensity.
    depth_props = regionprops_table(
        label_image=nuclei_labels,
        intensity_image=nuclei_depth_map,
        properties=["label", "intensity_mean"],
    )

    # Convert the dictionary output from regionprops_table to a pandas DataFrame.
    depth_df = pd.DataFrame(depth_props)

    # Rename the "intensity_mean" column to "depth" for clarity.
    depth_df.rename(columns={"intensity_mean": "depth"}, inplace=True)

    # Return the DataFrame, which now provides the mean depth value for each nucleus label.
    return depth_df