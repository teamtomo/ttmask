from pathlib import Path

import numpy as np
import einops
import typer
import mrcfile
from typing_extensions import Annotated


from ._cli import cli
from .soft_edge import add_soft_edge


@cli.command(name='tube')
def tube(
    sidelength: int = typer.Option(...),
    tube_height: float = typer.Option(...),
    tube_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("tube.mrc")),
):
    tube_radius = tube_diameter / 2

    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    difference = np.abs(positions - center)  # (100, 100, 100, 3)
    xy_distance = np.sum(difference[:, :, :, [1, 2]] ** 2, axis=-1) ** 0.5
    within_z = difference[:, :, :, 0] < (tube_height / (2 * pixel_size))
    within_xy = xy_distance < (tube_radius / pixel_size)


    mask[np.logical_and(within_z, within_xy)] = 1

    if wall_thickness != 0:
        within_xy_hollowing = xy_distance < ((tube_radius - wall_thickness) / pixel_size)
        mask[within_xy_hollowing] = 0

    mask = add_soft_edge(mask, soft_edge_width)

    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
