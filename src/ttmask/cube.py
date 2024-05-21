import numpy as np
import einops
import typer
from ._cli import cli
from scipy.ndimage import distance_transform_edt
import mrcfile


@cli.command(name='cube')
def cube(
    sidelength: int = typer.Option(...),
    cube_sidelength: float =typer.Option(...),
    soft_edge_width: float = typer.Option(0),
    mrc_voxel_size: float = typer.Option(...),
    output: str = typer.Option("cube.mrc")
):
    c = sidelength // 2
    center = np.array([c, c, c])
    mask = np.zeros(shape=(sidelength, sidelength, sidelength), dtype=np.float32)

    # 3d positions of all voxels
    positions = np.indices([sidelength, sidelength, sidelength])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    # calculate the distance between the center and every pixel position
    print(center.shape)
    print(positions.shape)

    print('calculating distance')
    difference = np.abs(positions - center)  # (100, 100, 100, 3)

    idx = np.all(difference < (np.array(cube_sidelength) / 2), axis=-1)

    mask[idx] = 1

    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= soft_edge_width) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / soft_edge_width) * np.pi

    mask[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)

    mrcfile.write(output, mask, voxel_size= mrc_voxel_size, overwrite=True)