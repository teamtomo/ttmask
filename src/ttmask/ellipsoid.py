from pathlib import Path

import numpy as np
import typer
import mrcfile
from typing import Tuple
from typing_extensions import Annotated

from .soft_edge import add_soft_edge
from ._cli import cli
from .box_setup import box_setup

def ellipsoid(
    sidelength: int, 
    ellipsoid_dimensions: Tuple[float, float, float],
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str
) -> np.ndarray:
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering)
    #converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    #extract xyz magnitudes from xyz_distances
    x_magnitude = xyz_distances[:, :, :, 2]
    y_magnitude = xyz_distances[:, :, :, 1]
    z_magnitude = xyz_distances[:, :, :, 0]

    #define desired dimensions in pixels (converting from angstrom)
    z_axis_length = ellipsoid_dimensions[0] / (2 * pixel_size)
    y_axis_length = ellipsoid_dimensions[1] / (2 * pixel_size)
    x_axis_length = ellipsoid_dimensions[2] / (2 * pixel_size)

    # set up criteria for which pixels are inside the ellipsoid and modify values to 1.
    in_ellipsoid = (((x_magnitude) ** 2) / (x_axis_length ** 2)) + ((y_magnitude ** 2) / (y_axis_length ** 2)) + (
        (z_magnitude ** 2) / (z_axis_length ** 2)) <= 1
    mask[in_ellipsoid] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        in_hollowing = (((x_magnitude) ** 2) / ((x_axis_length - wall_thickness) ** 2)) + (
                (y_magnitude ** 2) / ((y_axis_length - wall_thickness) ** 2)) + (
                           (z_magnitude ** 2) / ((z_axis_length - wall_thickness) ** 2)) <= 1
        mask[in_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='ellipsoid')
def ellipsoid_cli(
    sidelength: int = typer.Option(...),
    ellipsoid_dimensions: Annotated[Tuple[float, float, float], typer.Option()] = (None, None, None),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("ellipsoid.mrc")),
    centering: str = typer.Option("standard"),
):
    mask = ellipsoid(sidelength, ellipsoid_dimensions, wall_thickness, soft_edge_width, pixel_size, centering)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
