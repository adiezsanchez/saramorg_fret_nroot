from cellpose import models, core, io
import numpy as np

io.logger_setup()  # run this to get printing of progress

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
