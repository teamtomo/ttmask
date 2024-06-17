from pathlib import Path


import numpy as np
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup


@cli.command(name='sphere')
def sphere(
    sidelength: int = typer.Option(...),
    sphere_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("sphere.mrc")),
    wall_thickness: float = typer.Option(0),
):
    sphere_radius = sphere_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)

    #determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_centered)

    # set up criteria for which pixels are inside the sphere and modify values to 1.
    inside_sphere = distance_to_center < (sphere_radius / pixel_size)
    mask[inside_sphere] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        within_hollowing = distance_to_center < ((sphere_radius - wall_thickness) / pixel_size)
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    # output created with desired pixel size.
    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
