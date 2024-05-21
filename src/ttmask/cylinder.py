import numpy as np
import einops
import typer
from scipy.ndimage import distance_transform_edt
import mrcfile
from ._cli import cli


@cli.command(name='cylinder')
def cylinder(
    sidelength: int = typer.Option(...),
    cylinder_height: float = typer.Option(...),
    cylinder_outer_diameter: float = typer.Option(...),
    cylinder_inner_diameter: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    mrc_voxel_size: float = typer.Option(...),
    output: str = typer.Option("cylinder.mrc")
):
    cylinder_outer_radius = cylinder_outer_diameter / 2
    cylinder_inner_radius = cylinder_inner_diameter / 2

    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    print('calculating distance')
    difference = np.abs(positions - center)  # (100, 100, 100, 3)

    xy_distance = np.sum(difference[:, :, :, [1, 2]] ** 2, axis=-1) ** 0.5

    idx_z = difference[:, :, :, 0] < cylinder_height / 2
    idx_xy_outer = xy_distance < cylinder_outer_radius
    idx_xy_inner = xy_distance < cylinder_inner_radius

    mask[np.logical_and(idx_z, idx_xy_outer)] = 1
    mask[np.logical_and(idx_z, idx_xy_inner)] = 0

    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= soft_edge_width) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / soft_edge_width) * np.pi

    mask[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)

    mrcfile.write(output, mask, voxel_size= mrc_voxel_size, overwrite=True)

