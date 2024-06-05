from ._cli import cli
import numpy as np
import einops
import typer
from scipy.ndimage import distance_transform_edt
import mrcfile


@cli.command(name='sphere')
def sphere(
    sidelength: int = typer.Option(...),
    sphere_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output: str = typer.Option("sphere.mrc"),
    wall_thickness: float = typer.Option(0),
):
    sphere_radius = sphere_diameter / 2
    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    print(mask.dtype)

    # 2d positions of all pixels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    print('calculating distance')
    difference = np.abs(positions - center)  # (100, 100, 100, 3)
    distance = np.sum(difference ** 2, axis=-1) ** 0.5

    # calculate whether each pixel is inside or outside the circle
    print('calculating which pixels are in sphere')
    idx = distance < (sphere_radius / pixel_size)
    mask[idx] = 1

    if wall_thickness != 0:
        within_hollowing = distance < ((sphere_radius - wall_thickness) / pixel_size)
        mask[within_hollowing] = 0

    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= soft_edge_width) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / soft_edge_width) * np.pi

    mask[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)

    mrcfile.write(output, mask, voxel_size= pixel_size, overwrite=True)

