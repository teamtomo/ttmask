from pathlib import Path
from typing import Tuple
from typing_extensions import Annotated

import numpy as np
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup

def cuboid(
    sidelength: int, 
    cuboid_sidelengths: Tuple[float, float, float],
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple
) -> np.ndarray:
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)
    #converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # set up criteria for which pixels are inside the cuboid and modify values to 1.
    inside_cuboid = np.all(xyz_distances < (np.array(cuboid_sidelengths) / (2 * pixel_size)), axis=-1)
    mask[inside_cuboid] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        within_hollowing = np.all(xyz_distances < ((np.array(cuboid_sidelengths) / (2 * pixel_size)) - wall_thickness),
                                  axis=-1)
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)
    
    return mask

@cli.command(name='cuboid')
def cuboid_cli(
    sidelength: int = typer.Option(...),
    cuboid_sidelengths: Annotated[Tuple[float, float, float], typer.Option()] = (None, None, None),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cuboid.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50)
):
    mask = cuboid(sidelength, cuboid_sidelengths, wall_thickness, soft_edge_width, pixel_size, centering, center)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
