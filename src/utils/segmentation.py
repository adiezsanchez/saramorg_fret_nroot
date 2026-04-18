from cellpose import models, core, io
import numpy as np
from skimage.filters import difference_of_gaussians
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening, binary_erosion, disk, ball
from skimage.measure import label

io.logger_setup()  # run this to get printing of progress


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


# Check if notebook has GPU access
if core.use_gpu() == False:
    raise ImportError("No GPU access, change your runtime")

# Load CellposeSAM model
model = models.CellposeModel(gpu=True)

def predict_nuclei_labels(image: np.ndarray, rescale_factor: float, nuclei_channel: int) -> np.ndarray:
    """
    Predict nuclei labels using CellposeSAM using anisotropy correction.

    Args:
        image (np.ndarray): Image to predict nuclei labels from.
        rescale_factor (float): Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
        nuclei_channel (int): Channel index of the nuclei channel in the image.

    Returns:
        np.ndarray: Nuclei labels.
    """

    nuclei_labels, _ , _ = model.eval(image[nuclei_channel], do_3D=True, anisotropy=rescale_factor, z_axis=0, niter=1000)

    return nuclei_labels

def _normalize_percentile(image: np.ndarray, pmin: int = 1, pmax: int = 99) -> np.ndarray:
    """
    Normalize an image to the [0,1] range based on percentile clipping.

    Args:
        image (np.ndarray): The input image to be normalized.
        pmin (int, optional): Lower percentile for normalization. Defaults to 1.
        pmax (int, optional): Upper percentile for normalization. Defaults to 99.

    Returns:
        np.ndarray: The normalized image with values in [0, 1].
    """
    vmin, vmax = np.percentile(image, (pmin, pmax))
    image = np.clip(image, vmin, vmax)  # clip outlier intensities
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)  # rescale to [0, 1]
    else:
        image = image * 0  # avoid division by zero; returns zeros
    return image

def _normalize_full_range(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the [0,1] range using full 0-100 percentiles.

    Args:
        image (np.ndarray): The input image to be normalized.

    Returns:
        np.ndarray: The normalized image.
    """
    return _normalize_percentile(image, pmin=0, pmax=100)

def simulate_fluo_from_bf(
    lif_image: np.ndarray, 
    markers: tuple, 
    low_sigma: float = 1, 
    high_sigma: float = 2
) -> np.ndarray:
    """
    Simulate a fluorescent cell boundary channel from a brightfield image.

    Args:
        lif_image (np.ndarray): 4D image data, assumed shape (C, Z, Y, X) or (C, ...).
        markers (tuple): Channel descriptors linking channel names to indices.
        low_sigma (float, optional): Lower sigma for DoG. Defaults to 1.
        high_sigma (float, optional): Higher sigma for DoG. Defaults to 2.

    Returns:
        np.ndarray: Simulated boundary image for UNet3D input.
    """
    # Find the brightfield channel index in the user-defined markers
    bf_index = None
    for marker in markers:
        if marker[0].lower() in ["brightfield", "bf"]:
            bf_index = marker[1]
            break
    if bf_index is None:
        print("Please define the brightfield channel index under MARKERS.")

    # Normalize to remove outliers (and help later UNet3D inference)  
    brightfield_norm = _normalize_percentile(lif_image[bf_index])

    # Remove out of focus brightfield haze using Difference of Gaussians (DoG)
    bf_dog = difference_of_gaussians(brightfield_norm, low_sigma, high_sigma)

    # Normalize DoG response to [0, 1] before inversion
    bf_dog = _normalize_full_range(bf_dog)

    # Invert black and white values to simulate fluorescently labelled cell boundaries
    bf_inv = 1 - bf_dog

    return bf_inv

def generate_rough_root_3d(
    root_pmaps,
    nuclei_labels,
    probability_threshold=0.9,
    visualize=False,
    remove_nonconnected_components=False,
    viewer=None,
):
    """
    Generate a coarse 3D root mask by combining predictions from a root boundary segmentation map 
    and nuclei label masks. This function includes slice-wise morphological closing and hole filling 
    to produce contiguous masks even when contours are broken.

    Optionally, non-root, non-connected components can be removed: if `remove_nonconnected_components=True`,
    after all morphological operations, the mask is 3D-labeled, and only the largest connected 3D component is 
    retained. All other components are removed. This ensures the mask corresponds to the principal connected structure.

    Args:
        root_pmaps (np.ndarray): Probability map of root boundaries (shape: [1, Z, Y, X]).
                                 The array should be indexed at [0] to obtain the 3D map.
        nuclei_labels (np.ndarray): 3D array of nuclei instance labels (same shape as ZYX).
        probability_threshold (float, optional): Threshold value for converting the root boundary probability map 
            to a binary mask. Voxels with a probability above this value are considered part of the root boundary.
            Default is 0.9. Higher values will result in a more conservative root boundary mask.
        visualize (bool, optional): If True, add relevant masks as layers to the napari viewer. Default is False.
        remove_nonconnected_components (bool, optional): If True, retain only the largest 3D connected
            component of the mask. Default is False.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.

    Returns:
        np.ndarray: A boolean 3D mask representing the rough root segmentation. If 
            `remove_nonconnected_components=True`, only the largest connected component is retained.
    """
    # Threshold the root boundary probability map to create a binary mask
    root_boundary_mask = root_pmaps[0] > probability_threshold  # shape: (Z, Y, X)

    # Create a binary mask where nuclei are present
    nuclei_mask = nuclei_labels > 0  # shape: (Z, Y, X)

    # Combine the nuclei and root boundary masks as a starting point
    mask = (root_boundary_mask | nuclei_mask).astype(bool)

    # Preallocate the filled+closed output array
    filled_3d_closed = np.zeros_like(mask)

    # Perform slice-wise morphological closing and hole filling
    for z in range(mask.shape[0]):
        closed = binary_closing(mask[z], footprint=disk(5))           # Fill small holes/gaps in each slice
        filled_3d_closed[z] = binary_fill_holes(closed)               # Fill interior holes after closing

    if remove_nonconnected_components:
        # Label all 3D connected components
        labeled, num_labels = label(filled_3d_closed, return_num=True, connectivity=1)
        if num_labels > 0:
            # Retain only the largest 3D connected component
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # Ignore background
            largest_label = counts.argmax()
            filled_3d_closed = (labeled == largest_label)
        else:
            # No connected components found, return empty mask
            filled_3d_closed = np.zeros_like(filled_3d_closed, dtype=bool)

    # Optionally, visualize all relevant masks in napari
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_image(root_boundary_mask, name="root_boundary_mask", colormap="gray", blending="additive", opacity=1)
        v.add_image(nuclei_mask, name="nuclei_mask", colormap="gray", blending="additive", opacity=1)
        v.add_image(mask, name="combined_mask", colormap="gray", blending="additive", opacity=1)
        v.add_image(filled_3d_closed, name="closed+filled_2d_slice", colormap="orange", blending="additive", opacity=0.6)

    return filled_3d_closed

def _compute_core_3d_gated(core_2d_eroded, filled_3d_closed, slice_threshold):
    """
    Expand a 2D core into 3D only in slices where sufficient support exists.

    Args:
        core_2d_eroded (np.ndarray): 2D boolean core mask (Y, X)
        filled_3d_closed (np.ndarray): 3D boolean mask (Z, Y, X)
        slice_threshold (float): Minimum fraction of slice area overlapping core

    Returns:
        np.ndarray: 3D boolean mask (core_3d)
    """
    num_slices = filled_3d_closed.shape[0]
    core_3d = np.zeros_like(filled_3d_closed, dtype=bool)

    for z in range(num_slices):
        slice_mask = filled_3d_closed[z]

        slice_area = slice_mask.sum()
        if slice_area == 0:
            continue

        overlap = np.logical_and(slice_mask, core_2d_eroded).sum()

        # Compare to slice size (adaptive to thinning)
        ratio = overlap / slice_area

        if ratio >= slice_threshold:
            core_3d[z] = core_2d_eroded

    return core_3d

def fill_root_3d(
    filled_3d_closed,
    occupancy_threshold=0.9,
    erosion=3,
    slice_aware_filling=False,
    visualize=False,
    slice_threshold=0.3,
    viewer=None,
):
    """
    Refine a preliminary 3D root mask by robustly filling its interior (core) based on slice-by-slice occupancy.

    This function:
      - Calculates the occupancy of each (Y, X) voxel along the Z-axis (slices) to determine which 2D locations exist
        across the majority of slices (using occupancy_threshold).
      - Optionally erodes this 2D consensus "core" region using a disk of given radius (`erosion`), to remove edges
        and restrict core filling to highly central/root regions.
      - The eroded 2D core is then broadcast into 3D. If `slice_aware_filling` is True, core voxels are only added in slices with
        adequate local support (see `_compute_core_3d_gated`). Otherwise, the eroded core is applied to all slices.
      - The resulting 3D core is combined with the original mask to create a filled root, with edges preserved and inner gaps filled.

    Args:
        filled_3d_closed (np.ndarray): Boolean 3D array (Z, Y, X), preliminary root mask.
        occupancy_threshold (float, optional): Fraction (0-1). A (Y,X) pixel must be present in at least this fraction of slices (Z) to join the core. Default 0.9.
        erosion (int, optional): Radius (in pixels) for disk structural element for erosion of the 2D core. Default 3.
        slice_aware_filling (bool, optional): If True, only fills eroded core into slices with sufficient local support based on `slice_threshold` (adaptive filling). Default False.
        visualize (bool, optional): If True, adds occupancy map, 2D core, eroded core, and final filled mask to napari. Default False.
        slice_threshold (float, optional): Used *only* if `slice_aware_filling=True`. Minimum fraction of overlap required to fill core in a Z slice. Default 0.3.
        viewer (optional): Napari ``Viewer`` when ``visualize`` is True; see ``generate_rough_root_3d``.

    Returns:
        np.ndarray: Boolean 3D mask (same shape as input), with interior core (and gaps) robustly filled but boundaries preserved.
    """
    num_slices = filled_3d_closed.shape[0]

    # Compute per-pixel occupancy across Z
    occupancy = np.sum(filled_3d_closed, axis=0)  # shape: (Y, X)
    occupancy_norm = occupancy / num_slices

    # Threshold occupancy to define the 2D core present in sufficient slices
    threshold = int(np.ceil(occupancy_threshold * num_slices))
    core_2d = occupancy >= threshold

    # Erode the core to avoid overfilling edges
    core_2d_eroded = binary_erosion(core_2d, footprint=disk(erosion))

    # Expand eroded core back to 3D
    if slice_aware_filling:
        core_3d = _compute_core_3d_gated(core_2d_eroded, filled_3d_closed, slice_threshold)
    else:
        core_3d = np.repeat(core_2d_eroded[np.newaxis, ...], num_slices, axis=0)

    # Combine with the original to fill the core while preserving original edges
    filled_final = filled_3d_closed | core_3d

    # Optional napari visualization
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_image(occupancy_norm, name="occupancy_map")
        v.add_image(core_2d, name="core_2d", colormap="gray", blending="additive", opacity=1)
        v.add_image(core_2d_eroded, name="core_2d_eroded", colormap="gray", blending="additive", opacity=1)
        v.add_image(filled_final, name="filled_root_3d", colormap="yellow", blending="additive", opacity=0.6)

    return filled_final

def smooth_outer_root_surface_3d(
    filled_root_3d,
    erosion=5,
    smoothing=3,
    visualize=False,
    viewer=None,
):
    """
    Refine and smooth the outer surface of a 3D root mask by eroding the input mask 
    and then performing morphological closing and opening. 
    This procedure helps to remove small protrusions and irregularities on the surface,
    resulting in a smoother, more homogeneous root boundary.

    Args:
        filled_root_3d (np.ndarray): Boolean 3D mask (Z, Y, X) of the root structure.
        erosion (int, optional): Radius of ball structuring element for initial erosion. Larger values yield greater contraction of the root mask. Default is 5.
        smoothing (int, optional): Radius of ball structuring element applied to sequential closing and opening. Controls the degree of surface smoothing. Default is 3.
        visualize (bool, optional): If True, intermediate and final masks are shown in napari for inspection.
        viewer (optional): Napari ``Viewer`` when ``visualize`` is True; see ``generate_rough_root_3d``.

    Returns:
        np.ndarray: Boolean 3D mask representing the smoothed root (same shape as input).
    """
    # @root_segmentation.ipynb (13-2) -- Erode the root mask to eliminate thin surface structures and shrink mask slightly.
    eroded_root_3d = binary_erosion(filled_root_3d, footprint=ball(erosion))

    # @root_segmentation.ipynb (13-3,4) -- Smooth outer surface by closing small gaps, then remove small bulges with opening.
    smooth_mask = binary_closing(eroded_root_3d, ball(smoothing))
    smooth_mask = binary_opening(smooth_mask, ball(smoothing))

    # @root_segmentation.ipynb (13-6) -- Optionally visualize masks in napari viewer for user validation.
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_image(eroded_root_3d, name="eroded_root_3d", colormap="gray", blending="additive", opacity=0.8)
        v.add_image(smooth_mask, name="smooth_root_3d", colormap="green", blending="additive", opacity=0.5)

    return smooth_mask
