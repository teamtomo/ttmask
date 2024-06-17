from pathlib import Path

import numpy as np
import typer
import mrcfile

from .soft_edge import add_soft_edge
from ._cli import cli
from .box_setup import box_setup


@cli.command(name='cube')
def cube(
    sidelength: int = typer.Option(...),
    cube_sidelength: float = typer.Option(...),
    soft_edge_width: float = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cube.mrc")),
    wall_thickness: float = typer.Option(0),
):
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)
    #converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # set up criteria for which pixels are inside the cube and modify values to 1.
    in_cube = np.all(xyz_distances < np.array(cube_sidelength) / (pixel_size * 2), axis=-1)
    mask[in_cube] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        within_hollowing = np.all(xyz_distances < ((np.array(cube_sidelength) / (pixel_size * 2)) - wall_thickness),
                                  axis=-1)
        mask[within_hollowing] = 0

    #if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    #output created with desired pixel size.
    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
