import numpy as np
import einops


def box_setup(sidelength: int, centering: str, custom_center: tuple):
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
    coordinates = einops.rearrange(coordinates, 'zyx d h w -> d h w zyx')

    # coordinates now expressed relative to center
    coordinates_centered = coordinates - center
    return coordinates_centered, mask
