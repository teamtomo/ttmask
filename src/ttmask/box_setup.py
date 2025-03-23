"""Coordinate system setup for mask creation.

This module provides utilities for creating coordinate systems and empty masks
that are used as the basis for all mask creation functions.
"""

import einops
import numpy as np


def box_setup(
    sidelength: int, centering: str = "", custom_center: tuple = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Set up coordinate system and empty mask for mask generation.

    Creates a 3D coordinate system centered according to the
    specified method and an empty mask volume of the given sidelength.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic volume in pixels
    centering : str, optional
        Method for centering the coordinate system:
        - "standard": centered at the middle index of the volume
        - "visual": centered visually (adjusts for even sidelengths)
        - "custom": centered at custom_center coordinates
    custom_center : tuple, optional
        Custom (x,y,z) coordinates for the center when centering="custom"

    Returns
    -------
    tuple
        (coordinates_centered, mask)
        - the centered coordinate system and an empty mask
    """
    c = sidelength // 2
    if centering == "visual" and sidelength % 2 == 0:
        center = np.array([c, c, c]) - 0.5
    elif centering == "custom":
        center = np.array([custom_center])
    else:
        center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d coordinates of all voxels
    coordinates = np.indices([sidelength, sidelength, sidelength])
    coordinates = einops.rearrange(coordinates, "zyx d h w -> d h w zyx")

    # coordinates now expressed relative to center
    coordinates_centered = coordinates - center
    return coordinates_centered, mask
