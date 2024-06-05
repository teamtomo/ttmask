import numpy as np
import einops
import typer
from scipy.ndimage import distance_transform_edt
import mrcfile
from ._cli import cli
from .soft_edge import add_soft_edge


@cli.command(name='cylinder')
def cylinder(
    sidelength: int = typer.Option(...),
    cylinder_height: float = typer.Option(...),
    cylinder_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output: str = typer.Option("cylinder.mrc")
):
    cylinder_radius = cylinder_diameter / 2

    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    print('calculating distance')
    difference = np.abs(positions - center)  # (100, 100, 100, 3)
    xy_distance = np.sum(difference[:, :, :, [1, 2]] ** 2, axis=-1) ** 0.5
    within_z = difference[:, :, :, 0] < (cylinder_height / (2 * pixel_size))
    within_xy = xy_distance < (cylinder_radius / pixel_size)


    mask[np.logical_and(within_z, within_xy)] = 1

    if wall_thickness != 0:
        within_z_hollowing = difference[:, :, :, 0] < (cylinder_height - wall_thickness) / (2 * pixel_size)
        within_xy_hollowing = xy_distance < ((cylinder_radius - wall_thickness) / pixel_size)
        mask[np.logical_and(within_z_hollowing, within_xy_hollowing)] = 0

    mask = add_soft_edge(mask, soft_edge_width)

    mrcfile.write(output, mask, voxel_size=pixel_size, overwrite=True)
