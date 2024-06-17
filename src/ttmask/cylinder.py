from pathlib import Path

import numpy as np
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup


@cli.command(name='cylinder')
def cylinder(
    sidelength: int = typer.Option(...),
    cylinder_height: float = typer.Option(...),
    cylinder_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cylinder.mrc"))
):
    cylinder_radius = cylinder_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)

    # converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # set up criteria for which pixels are inside the cylinder and modify values to 1.

    xy_distance = np.linalg.norm(xyz_distances[:, :, :, [1, 2]], axis=-1)
    within_xy = xy_distance < (cylinder_radius / pixel_size)
    within_z = xyz_distances[:, :, :, 0] < (cylinder_height / (2 * pixel_size))
    mask[np.logical_and(within_z, within_xy)] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if wall_thickness != 0:
        within_z_hollowing = xyz_distances[:, :, :, 0] < (cylinder_height - wall_thickness) / (2 * pixel_size)
        within_xy_hollowing = xy_distance < ((cylinder_radius - wall_thickness) / pixel_size)
        mask[np.logical_and(within_z_hollowing, within_xy_hollowing)] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    # output created with desired pixel size.
    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
