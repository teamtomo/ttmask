from pathlib import Path
from typing import Tuple
from typing_extensions import Annotated

import numpy as np
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup

def sphere(
    sidelength: int, 
    sphere_diameter: float,
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple
) -> np.ndarray:
    sphere_radius = sphere_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask, _ = box_setup(sidelength, centering, center)

    # determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_centered, axis=-1)

    # set up criteria for which pixels are inside the sphere and modify values to 1.
    inside_sphere = distance_to_center < (sphere_radius / pixel_size)
    mask[inside_sphere] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        within_hollowing = distance_to_center < ((sphere_radius - wall_thickness) / pixel_size)
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='sphere')
def sphere_cli(
    sidelength: int = typer.Option(...),
    sphere_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("sphere.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50)
):
    mask = sphere(sidelength, sphere_diameter, wall_thickness, soft_edge_width, pixel_size, centering, center)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
