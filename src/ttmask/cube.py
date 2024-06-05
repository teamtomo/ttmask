import numpy as np
import einops
import typer
import mrcfile

from .soft_edge import add_soft_edge
from ._cli import cli


@cli.command(name='cube')
def cube(
    sidelength: int = typer.Option(...),
    cube_sidelength: float = typer.Option(...),
    soft_edge_width: float = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output: str = typer.Option("cube.mrc"),
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

    in_cube = np.all(difference < np.array(cube_sidelength) / (pixel_size * 2), axis=-1)
    mask[in_cube] = 1

    if wall_thickness != 0:
        within_hollowing = np.all(difference < ((np.array(cube_sidelength) / (pixel_size * 2)) - wall_thickness),
                                  axis=-1)
        mask[within_hollowing] = 0

    mask = add_soft_edge(mask, soft_edge_width)
    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
