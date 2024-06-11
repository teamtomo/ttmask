from pathlib import Path

import numpy as np
import einops
import typer
from typing import Tuple
from typing_extensions import Annotated
import mrcfile
from .soft_edge import add_soft_edge

from ._cli import cli

from ._cli import cli


@cli.command(name='cuboid')
def cuboid(
    sidelength: int = typer.Option(...),
    cuboid_sidelengths: Annotated[Tuple[float, float, float], typer.Option()] = (None, None, None),
    soft_edge_width: float = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: str = typer.Option(Path("cuboid.mrc")),
    wall_thickness: float = typer.Option(0),
):
    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([sidelength, sidelength, sidelength])
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

    inside_cuboid = np.all(difference < (np.array(cuboid_sidelengths) / (2 * pixel_size)), axis=-1)

    mask[inside_cuboid] = 1
       
    if wall_thickness != 0:
        within_hollowing = np.all(difference < ((np.array(cuboid_sidelengths) / (2 * pixel_size)) - wall_thickness), axis=-1)
        mask[within_hollowing] = 0
  
    mask = add_soft_edge(mask, soft_edge_width)

    mrcfile.write(output, mask, voxel_size= pixel_size, overwrite=True)

