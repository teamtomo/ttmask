import numpy as np
from scipy.ndimage import distance_transform_edt


def add_padding(mask: np.ndarray, width: float) -> np.ndarray:
    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= width) & (distance_from_edge != 0)
    output = np.copy(mask)
    output[boundary_pixels] = 1
    return output
