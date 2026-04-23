import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, regionprops_table
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


MORPHOLOGY_PROPERTIES = [
    "label",
    "centroid",
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

def _distance_map_from_root_mask(
    root_3d_mask: np.ndarray,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    spacing_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    """
    Compute distance-to-surface map in physical units (um).

    Args:
        root_3d_mask (np.ndarray): Boolean 3D mask (Z, Y, X) of the root body.
        pad_width (tuple): Padding configuration in np.pad format for (Z, Y, X).
        spacing_zyx_um (tuple[float, float, float]): Voxel spacing in um as (z, y, x).

    Returns:
        np.ndarray: Distance map in physical units with same shape as input.
    """
    mask_padded = np.pad(
        root_3d_mask.astype(bool),
        pad_width=pad_width,
        mode='constant',
        constant_values=0
    )
    dist_padded = distance_transform_edt(mask_padded, sampling=spacing_zyx_um)
    z0, _ = pad_width[0]
    y0, _ = pad_width[1]
    x0, _ = pad_width[2]
    return dist_padded[
        z0 : z0 + root_3d_mask.shape[0],
        y0 : y0 + root_3d_mask.shape[1],
        x0 : x0 + root_3d_mask.shape[2],
    ]

def _pad_half_root(
    root_3d_mask: np.ndarray,
    spacing_zyx_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Pads the root 3D mask with a border of zeros everywhere EXCEPT on the last Z slice (middle of the root).
    Pads Z only on the lower end (start), not on the upper end (stop); pads Y and X on both sides.
    Computes the distance transform on the padded mask and returns the distance map with padding removed.

    Args
        root_3d_mask (np.ndarray) Boolean 3D mask (Z, Y, X) of the root body.

    Returns
        np.ndarray Distance map with the same shape as the original mask, after padding and unpadding.
    """
    return _distance_map_from_root_mask(
        root_3d_mask,
        pad_width=((1, 0), (1, 1), (1, 1)),
        spacing_zyx_um=spacing_zyx_um,
    )

def _pad_full_root(
    root_3d_mask: np.ndarray,
    spacing_zyx_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
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
    return _distance_map_from_root_mask(
        root_3d_mask,
        pad_width=((1, 1), (1, 1), (1, 1)),
        spacing_zyx_um=spacing_zyx_um,
    )

def calculate_distance_to_root_surface(
    nuclei_labels: np.ndarray,
    root_3d_mask: np.ndarray,
    pad_full_root: bool = False,
    spacing_zyx_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
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
        spacing_zyx_um (tuple[float, float, float], optional): Voxel spacing in um
            ordered as (z, y, x). Defaults to isotropic (1.0, 1.0, 1.0).
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
        dist_map = _pad_full_root(root_3d_mask, spacing_zyx_um=spacing_zyx_um)
        is_flooded = False
        flooded_planes = []
    else:
        # Compensate for truncated tails before half-root padding.
        root_3d_mask, is_flooded, flooded_planes = _flood_fill_planes_below_threshold(
            root_3d_mask
        )
        dist_map = _pad_half_root(root_3d_mask, spacing_zyx_um=spacing_zyx_um)

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
    markers: list[tuple[str, int, str]],
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
        markers (list[tuple[str, int, str]]): Marker definitions as
            ``(marker_name, channel_index, marker_role_or_descriptor)``.
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
    for marker_name, ch_nr, *_ in markers:
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

def classify_root_cap_nuclei(
    props_df: "pd.DataFrame",
    feature_columns: list = ['edCitrine_CTRL_mean_int', 'area'],
    weights: list = [1, 1]
) -> "pd.DataFrame":
    """
    Classify nuclei as belonging to the root cap or the rest of the root structure using clustering.

    This function divides nuclei into two groups (root cap vs. root) based on user-selected features and their respective weights.
    By default, it uses:
      - 'edCitrine_CTRL_mean_int' (higher for root cap)
      - 'area' (smaller for root cap)

    Clustering is performed using KMeans on standardized features.

    Args:
        props_df (pd.DataFrame): DataFrame containing per-nucleus features.
        feature_columns (list, optional): List of feature column names to use for clustering. Default is ['edCitrine_CTRL_mean_int', 'area'].
        weights (list or np.ndarray, optional): List or array of weights for each feature in feature_columns. Default is [1, 1].

    Returns:
        pd.DataFrame: The input DataFrame with additional columns:
            - 'tip_cluster_id': cluster assignment (0 or 1) for each nucleus
            - 'root_part': "root" or "root_cap" assignment per nucleus

    Raises:
        AssertionError: If the nuclei cluster mapped to "root_cap" does not have a higher 
            mean value of the first feature (feature_columns[0]) than the cluster mapped to "root".

    Notes:
        - Features are scaled before clustering for equal weighting unless otherwise specified.
        - The assignment "root_cap" is mapped to the cluster with higher mean on the first feature.
        - Make sure feature_columns and weights are the same length.
    """

    assert len(feature_columns) == len(weights), "feature_columns and weights must be of same length"

    # Select features specified by the user
    features = props_df[feature_columns].values

    # Scale features for balanced clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply weights to each feature (element-wise multiplication)
    weights_array = np.array(weights)
    features_weighted = features_scaled * weights_array

    # Perform k-means clustering (2 clusters: root_body, root_cap)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features_weighted)

    # Add cluster assignments to DataFrame
    props_df['tip_cluster_id'] = cluster_ids

    # Map cluster IDs to "root_body" and "root_cap" based on mean of first feature
    cluster_means = (
        props_df.groupby('tip_cluster_id')[feature_columns[0]]
        .mean()
        .sort_values()
    )
    ordered_cluster_ids = cluster_means.index.tolist()
    tissue_layer_map = dict(zip(ordered_cluster_ids, ["root_body", "root_cap"]))
    props_df['root_part'] = props_df['tip_cluster_id'].map(tissue_layer_map)

    # Sanity check: root_cap must have higher feature_columns[0] value than root
    mean_root_cap = props_df.loc[props_df['root_part'] == 'root_cap', feature_columns[0]].mean()
    mean_root = props_df.loc[props_df['root_part'] == 'root_body', feature_columns[0]].mean()
    assert mean_root_cap > mean_root, (
        f"Sanity check failed: root_cap mean {feature_columns[0]} is not higher than root_body"
    )

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

from typing import Optional

def compute_fret_ratios(
    df: "pd.DataFrame",
    markers: Optional[list[tuple[str, int, str]]] = None
) -> "pd.DataFrame":
    """
    Compute FRET ratios (raw and normalized) from per-nucleus intensity features using a configurable markers list.

    FRET ratio = (donor-excited acceptor emission) / (donor-excited donor emission)
                = (DA marker) / (DD marker)

    This function identifies the donor (DD) and donor-excited acceptor (DA) channels from the provided markers
    list of tuples, extracts the corresponding intensity columns from the DataFrame, and computes the FRET ratios.

    Note:
    - Saturated pixels are not excluded from the FRET ratio calculation.
    - No bleed-through or direct excitation correction is applied in this raw ratio.
    - Normalization scope is per image.

    Args:
        df (pd.DataFrame): Per-nucleus feature table containing intensity columns for all markers.
        markers (Optional[list[tuple[str, int, str]]]): Marker tuples in the form
            ``(marker_name, channel_index, role_str)``. The role string must be in
            position 3 of each tuple and should contain ``"DD"`` or ``"DA"``.

    Returns:
        pd.DataFrame: The same DataFrame with new FRET ratio columns appended.
    """

    # Set default markers if none provided
    if markers is None:
        markers = [
            ("edCerulean_CTRL", 0, "DD"),
            ("edCitrine_FRET", 1, "DA"),
            ("edCitrine_CTRL", 2, "DD"),
            ("brightfield", 3, "root_structure")
        ]

    # Helper to scan the marker tuples for "DD" and "DA" channels (case-insensitive) by role_str
    dd_candidates = [(name, role) for name, _, role, *_ in markers if "dd" in role.lower()]
    da_candidates = [(name, role) for name, _, role, *_ in markers if "da" in role.lower()]
    if not dd_candidates or not da_candidates:
        raise ValueError("Could not identify DD and DA markers from markers list.")

    # Take first DD and DA markers found
    dd_marker_name, _ = dd_candidates[0]
    da_marker_name, _ = da_candidates[0]

    # Get corresponding columns from the DataFrame for sum and mean intensities
    dd_sum = df[f"{dd_marker_name}_sum_int"]
    da_sum = df[f"{da_marker_name}_sum_int"]
    dd_mean = df[f"{dd_marker_name}_mean_int"]
    da_mean = df[f"{da_marker_name}_mean_int"]

    # Compute mask for valid signals (avoid division by zero)
    valid_sum = (dd_sum > 0) & (da_sum > 0)
    valid_mean = (dd_mean > 0) & (da_mean > 0)

    # Compute raw FRET ratios (set to NaN if invalid)
    df["FRET_ratio_sum"] = np.where(valid_sum, da_sum / dd_sum, np.nan)
    df["FRET_ratio_mean"] = np.where(valid_mean, da_mean / dd_mean, np.nan)

    # Helper function to normalize series between 0 and 1
    def normalize(series: "pd.Series") -> "pd.Series":
        min_val = series.min(skipna=True)
        max_val = series.max(skipna=True)
        if max_val == min_val:
            # Avoid division by zero if constant or all NaN
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val)

    # Compute normalized FRET ratios (range: 0 to 1)
    df["FRET_ratio_sum_norm_per_image"] = normalize(df["FRET_ratio_sum"])
    df["FRET_ratio_mean_norm_per_image"] = normalize(df["FRET_ratio_mean"])

    return df

def map_root_body_depth_clusters_to_tissue_layers(
    props_df: "pd.DataFrame"
) -> "pd.DataFrame":
    """
    Cluster root body nuclei by normalized depth and assign tissue layer identity.

    This function excludes nuclei annotated as 'root_cap', clusters the remaining (root body)
    nuclei based on their depth using k-means (n_clusters=5), and deterministically maps the
    resulting clusters to sequential cluster IDs and canonical tissue layer names. The output dataframe
    maps each nucleus (label) to:
        - depth_cluster_id (int): Deterministic cluster index (2...6 for root body, 1 for root cap),
        - tissue_layer (str): Biological layer name ("Epi", "Cor", "End", "Per", "Vasc" for root body, "root_cap" for root cap).

    Args:
        props_df (pd.DataFrame): Nucleus-level feature table. Must include columns:
            - "label": unique nucleus IDs,
            - "root_part": string annotation ("root_body" or "root_cap"),
            - "depth": float, normalized or absolute depth for each nucleus.

    Returns:
        pd.DataFrame: Mini-table (with columns: "label", "depth_cluster_id", "tissue_layer")
            containing only root_body nuclei; can be merged/joined back to props_df.
    """
    # Filter out root cap nuclei before performing depth clustering
    # Create a minimal copy of the original dataframe
    filtered_df: "pd.DataFrame" = props_df.loc[
        props_df["root_part"] == "root_body", ["depth", "label"]
    ].copy()

    # Perform k-means clustering (5 groups) on the 'depth' column
    depth_values: "np.ndarray" = filtered_df["depth"].to_numpy().reshape(-1, 1)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    filtered_df["cluster_raw"] = kmeans.fit_predict(depth_values)  # arbitrary labels 0..4

    # Compute mean depth per raw cluster and sort shallow -> deep
    cluster_order = (
        filtered_df.groupby("cluster_raw")["depth"]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    # Build deterministic remaps with ordered numeric ids (e.g., 2..6). root_cap will become 1 later
    raw_to_depth_id = {raw: i + 2 for i, raw in enumerate(cluster_order)}
    filtered_df["depth_cluster_id"] = filtered_df["cluster_raw"].map(raw_to_depth_id)

    # Assign tissue layers in the same ordered direction
    layer_names = ["Epi", "Cor", "End", "Per", "Vasc"]  # shallow -> deep
    raw_to_layer = {raw: layer_names[i] for i, raw in enumerate(cluster_order)}
    filtered_df["tissue_layer"] = filtered_df["cluster_raw"].map(raw_to_layer)

    return filtered_df[["label", "depth_cluster_id", "tissue_layer"]]

def merge_root_cap_into_tissue_layers(
    props_df: "pd.DataFrame",
    filtered_df: "pd.DataFrame"
) -> "pd.DataFrame":
    """
    Merge depth cluster/layer assignments into the full nucleus feature DataFrame,
    handling root cap nuclei by defaulting missing values to canonical root cap assignments.

    This function merges tissue layer and depth cluster assignments (from `filtered_df`)
    into the full nucleus table (`props_df`). For nuclei annotated as root cap
    ("root_part" == "root_cap"), it assigns:
        - "tissue_layer" = "root_cap"
        - "depth_cluster_id" = 1

    For all other nuclei, values from filtered_df are used; any missing cluster
    assignments (should be rare) are assigned to 0 as a fallback.

    Args:
        props_df (pd.DataFrame): Full per-nucleus feature table. Must include columns:
            - "label": unique nucleus IDs
            - "root_part": "root_body" or "root_cap"
        filtered_df (pd.DataFrame): DataFrame containing at least:
            - "label": nucleus IDs for root_body nuclei
            - "depth_cluster_id": depth cluster indices
            - "tissue_layer": tissue layer names

    Returns:
        pd.DataFrame: `props_df` with "depth_cluster_id" and "tissue_layer" columns
            assigned for both root_body and root_cap nuclei.
    """
    # Merge filtered_df into props_df on 'label'
    props_df = props_df.merge(
        filtered_df[['label', 'depth_cluster_id', 'tissue_layer']],
        on='label',
        how='left'
    )

    # Default missing tissue layer to root_cap
    props_df['tissue_layer'] = props_df['tissue_layer'].fillna('root_cap')

    # Start with nullable int so we can assign selectively
    props_df['depth_cluster_id'] = props_df['depth_cluster_id'].astype('Int64')

    # Force root_cap rows to cluster id 1
    root_cap_mask = props_df['root_part'].eq('root_cap')
    props_df.loc[root_cap_mask, 'depth_cluster_id'] = 1

    # For any remaining missing values (non-root_cap), choose fallback (e.g., 0)
    props_df['depth_cluster_id'] = props_df['depth_cluster_id'].fillna(0).astype(int)

    # Keep tissue_layer consistent for root_cap
    props_df.loc[root_cap_mask, 'tissue_layer'] = 'root_cap'

    # Sanity checks
    assert (
        props_df.loc[props_df["tissue_layer"] == "root_cap", "root_part"] == "root_cap"
    ).all(), "Sanity check failed: Some 'root_cap' tissue_layer rows are not root_part == 'root_cap'"

    assert (
        props_df.loc[props_df["root_part"] == "root_cap", "depth_cluster_id"] == 1
    ).all(), "Sanity check failed: Some root_cap rows do not have depth_cluster_id == 1"

    return props_df