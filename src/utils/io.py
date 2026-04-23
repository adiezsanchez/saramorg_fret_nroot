from pathlib import Path
import liffile
import numpy as np
import tifffile


def list_containers(directory_path: str, file_format: str) -> list:
    """
    List all image files in a given directory with the specified format (extension).

    Args:
        directory_path (str): Path to the directory to search for image files.
        file_format (str): File extension (without the dot), e.g. "tif", "png".

    Returns:
        list: List of image file paths as strings.
    """
    images = []

    for file_path in sorted(
        Path(directory_path).glob(f"*.{file_format}"),
        key=lambda p: p.name.lower(),
    ):
        images.append(str(file_path))

    return images

def explore_lif_container(file_path: str, display: bool = False) -> tuple[int, str]:
    """
    Explore a .lif container file and print image metadata (name, dimensions and shape) inside the container.
    This function does not return the container object, it only returns the number of images inside the container.

    Args:
        file_path (str): Path to the .lif container file.                         
        display (bool): If True image metadata (name, dimensions and shape) inside the container will be printed.

    Returns:
        tuple:
            - nr_imgs (int): Number of images inside the container.
            - lif_container_id (str): Name of the .lif container file.
    """
    # Extract lif_container filename
    lif_container_id = Path(file_path).stem
    
    # Read a single .lif container
    with liffile.LifFile(file_path) as lif_container:

        if display:

            # List all images inside the container
            for img in lif_container.images:
                print(f"Image name: {img.name}, Dimensions: {img.dims}, Array Shape: {img.shape}")

        # Store number of images inside the container
        nr_imgs = len(lif_container.images)

    return nr_imgs, lif_container_id

def load_lif_image(file_path: str, image_index: int) -> tuple["np.ndarray", str, object]:
    """
    Return a specific image from a Leica .lif file as a NumPy array, along with its name and XML metadata.

    Args:
        file_path (str): Path to the .lif container file.
        image_index (int): Index of the image to load within the container.

    Returns:
        tuple:
            - img (np.ndarray): The image data array with shape (C, Z, Y, X).
            - img_name (str): The name of the image.
            - xml_metadata (object): The XML metadata associated with the image.
    """
    with liffile.LifFile(file_path) as lif_container:

        # Get the image object at position [image_index] inside the container
        img_obj = lif_container.images[image_index]

        # Return the image at position [index] inside the container as a numpy array
        img = img_obj.asarray()

        # Transpose to return C, Z, Y, X
        img = img.transpose(1,0,2,3)

        # Extract the image name
        img_name = img_obj.name

        # Extract XML metadata
        xml_metadata = img_obj.xml_element

    return img, img_name, xml_metadata 

def ensure_output_dir(
    base_output_dir: str | Path,
    lif_container_id: str,
    results_type: str
) -> Path:
    """
    Create and return the output directory used to store np.array results for one .lif container and results type.

    Args:
        base_output_dir (str | Path): Base output directory.
        lif_container_id (str): Name of the .lif container without extension.
        results_type (str): Subdirectory name indicating the type of results being stored (e.g., "nuclei_labels").

    Returns:
        Path: Path to the output directory for the specified results type and .lif container.
    """
    results_dir = Path(base_output_dir) / results_type / lif_container_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Could not create {results_type} results directory: {results_dir}")

    return results_dir

def load_precomputed_results_if_available(results_dir: str | Path, image_id: str, results_type: str) -> np.ndarray | None:
    """
    Load precomputed np.array results for one image if they are already stored on disk (as .tif).

    Args:
        labels_dir (str | Path): Directory where nuclei labels are stored.
        image_id (str): Name of the image.
        results_type (str): Precomputed results being loaded, match results_type from ensure_output_directory.
        (e.g. "nuclei_labels", "root_mask", "depth_map")

    Returns:
        np.ndarray | None: Loaded results when available, otherwise None.
    """
    precomputed_results_path = Path(results_dir) / f"{image_id}_{results_type}.tif"

    if not precomputed_results_path.is_file():
        return None

    return tifffile.imread(precomputed_results_path)

def _extract_pixel_sizes_um(xml_element) -> tuple[float, float, float]:
    """
    Extract (x_um, y_um, z_um) from Leica XML metadata.
    Uses DimensionDescription entries:
      - DimID 1: X
      - DimID 2: Y
      - DimID 3: Z
    Assumes Unit is meters ('m').
    """

    metadata_dict = liffile.xml2dict(xml_element)

    dims = (
        metadata_dict["Element"]["Data"]["Image"]["ImageDescription"]["Dimensions"]["DimensionDescription"]
    )

    by_id = {d["DimID"]: d for d in dims}
    x = by_id[1]
    y = by_id[2]
    z = by_id[3]

    # meters -> micrometers
    m_to_um = 1e6

    x_um = (x["Length"] / x["NumberOfElements"]) * m_to_um
    y_um = (y["Length"] / y["NumberOfElements"]) * m_to_um

    # Z spacing is between planes, so divide by (N-1)
    z_um = (z["Length"] / (z["NumberOfElements"] - 1)) * m_to_um

    return x_um, y_um, z_um

def get_voxel_spacing_zyx_um(xml_element) -> tuple[float, float, float]:
    """
    Return voxel spacing in micrometers as (z_um, y_um, x_um).

    Args:
        xml_element: Parsed XML metadata element containing pixel size information.

    Returns:
        tuple[float, float, float]: Voxel spacing ordered as (z, y, x) in um.
    """
    x_um, y_um, z_um = _extract_pixel_sizes_um(xml_element)
    return z_um, y_um, x_um

def calculate_rescale_factor(xml_element, display: bool = False) -> float:
    """
    Calculate the anisotropy rescale factor along the Z-axis based on pixel sizes in the Leica XML metadata.

    The returned value is the ratio of Z-resolution (z_um) to the average XY-resolution (mean of x_um and y_um).
    This is useful for rescaling volumetric data so that voxel dimensions are isotropic.

    Args:
        xml_element: Parsed XML metadata element containing pixel size information.

    Returns:
        float: Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
    """
    x_um, y_um, z_um = _extract_pixel_sizes_um(xml_element)

    # Calculate anisotropy to rescale across the Z-axis (ratio of Z-resolution to XY-resolution)
    rescale_factor = z_um / np.mean([x_um , y_um])

    if display:
        print(f"x_um: {x_um}, y_um: {y_um}, z_um: {z_um}")
        print(f"Rescale factor: {rescale_factor}")

    return rescale_factor