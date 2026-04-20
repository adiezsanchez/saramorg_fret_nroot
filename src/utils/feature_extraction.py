import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

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
) -> np.ndarray:
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
        np.ndarray: 3D array (same shape as input) with normalized per-nucleus depth values; zero outside labeled regions.
    """
    # Choose padding strategy depending on root type (full or half cone)
    if pad_full_root:
        dist_map = _pad_full_root(root_3d_mask)
    else:
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

    return depth_image