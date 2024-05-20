import numpy as np
import einops
import typer
from typing import Tuple
from ._cli import cli
from typing_extensions import Annotated
from scipy.ndimage import distance_transform_edt
import mrcfile


@cli.command(name='cuboid')
def cuboid(
    boxsize: int = typer.Option(...),
    cuboid_sidelengths: Annotated[Tuple[float, float, float], typer.Option()] = (None, None, None),
    soft_edge_size: float = typer.Option(...),
    mrc_voxel_size: float = typer.Option(...),
):
    c = boxsize // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(boxsize, boxsize, boxsize), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([boxsize, boxsize, boxsize])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    # calculate the distance between the center and every pixel position
    print(center.shape)
    print(positions.shape)

    print('calculating distance')
    difference = np.abs(positions - center)  # (100, 100, 100, 3)
    # z = difference[:, :, :, 0]
    # y = difference[:, :, :, 1]
    # x = difference[:, :, :, 2]
    # idx_z = z < cuboid_sidelengths[0] / 2
    # idx_y = y < cuboid_sidelengths[1] / 2
    # idx_x = x < cuboid_sidelengths[2] / 2

    # mask[np.logical_not(idx)] = 1 #if you wanted to do opposite for whatever reason

    idx = np.all(difference < (np.array(cuboid_sidelengths) / 2), axis=-1)

    mask[idx] = 1

    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= soft_edge_size) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / soft_edge_size) * np.pi

    mask[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)

    mrcfile.write("cuboid.mrc", mask, voxel_size= mrc_voxel_size, overwrite=True)
