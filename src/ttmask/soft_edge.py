import numpy as np
from scipy.ndimage import distance_transform_edt


def add_soft_edge(mask: np.ndarray, width: float) -> np.ndarray:
    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= width) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / width) * np.pi
    output = np.copy(mask)
    output[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)
    return output
