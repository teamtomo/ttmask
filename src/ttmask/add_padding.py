"""Padding utilities for masks.

This module provides functions for adding padding to binary masks,
which can be useful for expanding mask regions.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def add_padding(mask: np.ndarray, width: float) -> np.ndarray:
    """Add padding to a binary mask.

    Expands the mask by a specified width, setting all voxels within the
    specified distance from the original mask to 1.

    Parameters
    ----------
    mask : np.ndarray
        The input binary mask (values 0 and 1)
    width : float
        Width of the padding in pixels

    Returns
    -------
    np.ndarray
        The padded mask
    """
    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= width) & (distance_from_edge != 0)
    output = np.copy(mask)
    output[boundary_pixels] = 1
    return output
