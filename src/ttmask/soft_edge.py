"""Soft edge generation for masks.

This module provides utilities for adding soft edges to binary masks,
which can be useful for preventing artifacts in processing.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def add_soft_edge(mask: np.ndarray, width: int) -> np.ndarray:
    """Add a soft edge to a binary mask.

    Creates a gradual transition from 1 to 0 at the edge of a binary mask
    using a cosine function to achieve a smooth falloff.

    Parameters
    ----------
    mask : np.ndarray
        The input binary mask (values 0 and 1)
    width : int
        Width of the soft edge in pixels

    Returns
    -------
    np.ndarray
        The mask with added soft edges
    """
    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= width) & (distance_from_edge != 0)
    normalised_distance_from_edge = (
        distance_from_edge[boundary_pixels] / width
    ) * np.pi
    output = np.copy(mask)
    output[boundary_pixels] = 0.5 * np.cos(normalised_distance_from_edge) + 0.5
    return output
