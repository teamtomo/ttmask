from pathlib import Path
from typing import Tuple
from typing_extensions import Annotated

import numpy as np
import typer
import mrcfile


from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup

def tube(
    sidelength: int, 
    tube_height: float, 
    tube_diameter: float, 
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple
) -> np.ndarray:
    tube_radius = tube_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)

    #converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # set up criteria for which pixels are inside the tube and modify values to 1.
    xy_distance = np.linalg.norm(xyz_distances[:, :, :, [1, 2]], axis=-1)
    within_z = xyz_distances[:, :, :, 0] < (tube_height / (2 * pixel_size))
    within_xy = xy_distance < (tube_radius / pixel_size)
    mask[np.logical_and(within_z, within_xy)] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        within_xy_hollowing = xy_distance < ((tube_radius - wall_thickness) / pixel_size)
        mask[within_xy_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='tube')
def tube_cli(
    sidelength: int = typer.Option(...),
    tube_height: float = typer.Option(...),
    tube_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("tube.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50)
):
    mask = tube(sidelength, tube_height, tube_diameter, wall_thickness, soft_edge_width, pixel_size, centering, center)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
