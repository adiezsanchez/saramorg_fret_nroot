from pathlib import Path
import liffile


def list_images(directory_path: str, file_format: str) -> list:
    """
    List all image files in a given directory with the specified format (extension).

    Args:
        directory_path (str): Path to the directory to search for image files.
        file_format (str): File extension (without the dot), e.g. "tif", "png".

    Returns:
        list: List of image file paths as strings.
    """
    images = []

    for file_path in Path(directory_path).glob(f"*.{file_format}"):
        images.append(str(file_path))
        
    return images

def extract_pixel_sizes_um(xml_element) -> tuple[float, float, float]:
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