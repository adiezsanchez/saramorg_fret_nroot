from pathlib import Path
import liffile


def list_images (directory_path, format):

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    for file_path in directory_path.glob(f"*.{format}"):
        images.append(str(file_path))

    return images

def read_image (image, slicing_factor_xy, log=True):
    """Read raw image microscope files (.lif), apply downsampling if needed and return filename and a numpy array"""

    # Read path storing raw image and extract filename
    file_path = Path(image)
    filename = file_path.stem

    # Extract file extension
    extension = file_path.suffix

    if extension == ".lif":
        # Read stack from .nd2 (z, ch, x, y) or (ch, x, y)
        img = liffile.imread(image)
        
    else:
        print ("Implement new file reader")

    # Apply slicing trick to reduce image size (xy resolution)
    try:
        img = img[:, ::slicing_factor_xy, ::slicing_factor_xy]
    except IndexError as e:
        print(f"Slicing Error: {e}")
        print(f"Slicing parameters: Slicing_XY:{slicing_factor_xy}")

    if log:
        # Feedback for researcher
        print(f"\nImage analyzed: {filename}")
        #print(f"Original Array shape: {img.shape}")
        #print(f"Compressed Array shape: {img.shape}")

    return img, filename