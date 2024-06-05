from ._cli import cli
import numpy as np
import einops
import typer
from scipy.ndimage import distance_transform_edt
import mrcfile
from .soft_edge import soft_edge


@cli.command(name='ellipsoid')
def ellipsoid(
    sidelength: int = typer.Option(...),
    width: float = typer.Option(...),
    height: float = typer.Option(...),
    depth: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output: str = typer.Option("ellipsoid.mrc"),
):
    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # positions of all pixels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    difference = np.abs(positions - center)  # (100, 100, 100, 3)
    distances = np.linalg.norm(difference, axis=-1)

    x_magnitude = difference[:, :, :, 2]
    y_magnitude = difference[:, :, :, 1]
    z_magnitude = difference[:, :, :, 0]

    z_axis_length = depth / (2 * pixel_size)
    y_axis_length = height / (2 * pixel_size)
    x_axis_length = width / (2 * pixel_size)

    in_ellipsoid = (((x_magnitude) ** 2) / (x_axis_length ** 2)) + ((y_magnitude ** 2) / (y_axis_length ** 2)) + (
            (z_magnitude ** 2) / (z_axis_length ** 2)) <= 1
    mask[in_ellipsoid] = 1

    soft_edge(mask, soft_edge_width)

    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
