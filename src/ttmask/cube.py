from pathlib import Path
import numpy as np
import typer
import mrcfile

from .soft_edge import add_soft_edge
from ._cli import cli
from .box_setup import box_setup

def cube(
    sidelength: int, 
    cube_sidelength: float,
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str
) -> np.ndarray:
     # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering)
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

    return mask

@cli.command(name='cube')
def cube_cli(
    sidelength: int = typer.Option(...),
    cube_sidelength: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cube.mrc")),
    centering: float = typer.Option("standard"),
):
    mask = cube(sidelength, cube_sidelength, wall_thickness, soft_edge_width, pixel_size, centering)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
