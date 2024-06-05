from pathlib import Path

import numpy as np
import einops
import typer
from scipy.ndimage import distance_transform_edt
import mrcfile
from ._cli import cli


@cli.command(name='cone')
def cone(
    sidelength: int = typer.Option(...),
    cone_height: float = typer.Option(...),
    cone_base_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output: Path = typer.Option(Path("cone.mrc"))
):
    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    centered = positions - center  #pixels relative to center point
    magnitudes = np.linalg.norm(centered, axis=-1)

    magnitudes = einops.rearrange(magnitudes, 'd h w -> d h w 1')

    # Check for zeros in magnitudes and replace them with a small value to avoid Nan warning
    near_zero = 1e-8
    magnitudes = np.where(magnitudes == 0, near_zero, magnitudes)
    normalised = centered / magnitudes

    principal_axis = np.array([1, 0, 0])
    dot_product = np.dot(normalised, principal_axis)
    angles_radians = np.arccos(dot_product)
    angles = np.rad2deg(angles_radians)

    z_distance = centered[:, :, :, 0]  # (100, 100, 100)

    # Calculate the angle from the tip of the cone to the edge of the base
    cone_base_radius = (cone_base_diameter / 2) / pixel_size
    cone_angle = np.rad2deg(np.arctan(cone_base_radius / cone_height))

    within_cone_height = z_distance < (cone_height / pixel_size)
    within_cone_angle = angles < cone_angle

    # mask[within_cone_height] = 1
    mask[np.logical_and(within_cone_height, within_cone_angle)] = 1

    # Shift the mask in the z-axis by cone_height / 2
    z_shift = -int(cone_height / 2)
    mask = np.roll(mask, z_shift, axis=0)


    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= soft_edge_width) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / soft_edge_width) * np.pi

    mask[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)


    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)


