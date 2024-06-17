from pathlib import Path

import numpy as np
import einops
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup


@cli.command(name='cone')
def cone(
    sidelength: int = typer.Option(...),
    cone_height: float = typer.Option(...),
    cone_base_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cone.mrc"))
):
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)
    # distances between each pixel and center :
    magnitudes = np.linalg.norm(coordinates_centered, axis=-1)
    magnitudes = einops.rearrange(magnitudes, 'd h w -> d h w 1')

    # Check for zeros in magnitudes and replace them with a small value to avoid Nan warning
    near_zero = 1e-8
    magnitudes = np.where(magnitudes == 0, near_zero, magnitudes)
    normalised = coordinates_centered / magnitudes

    principal_axis = np.array([1, 0, 0])
    dot_product = np.dot(normalised, principal_axis)
    angles_radians = np.arccos(dot_product)
    angles = np.rad2deg(angles_radians)

    z_distance = coordinates_centered[:, :, :, 0]  # (100, 100, 100)

    # Calculate the angle from the tip of the cone to the edge of the base
    cone_base_radius = (cone_base_diameter / 2) / pixel_size
    cone_angle = np.rad2deg(np.arctan(cone_base_radius / cone_height))

    within_cone_height = z_distance < (cone_height / pixel_size)
    within_cone_angle = angles < cone_angle

    # mask[within_cone_height] = 1
    mask[np.logical_and(within_cone_height, within_cone_angle)] = 1

    #   need to adjust the center of the hollow cone otherwise the cone thins towards the apex
    # thickness will need to decrease towards the apex anyway - it's surely not possible / realistic?

    # Shift the mask in the z-axis by cone_height / 2
    z_shift = -int(cone_height / 2)
    mask = np.roll(mask, z_shift, axis=0)
    mask = add_soft_edge(mask, soft_edge_width)

    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
