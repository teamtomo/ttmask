from ._cli import cli
import numpy as np
import einops
import napari
import typer
from scipy.ndimage import distance_transform_edt

@cli.command(name='sphere')
def sphere(
    boxsize: int = typer.Option(...),
    sphere_diameter: float = typer.Option(...),
    soft_edge_size: int = typer.Option(...),
):

    sphere_radius = sphere_diameter / 2

    center = np.array([boxsize // 2, boxsize // 2,
                       boxsize // 2])
    mask = np.zeros(shape=(boxsize, boxsize, boxsize))

    # 2d positions of all pixels
    positions = np.indices([boxsize, boxsize, boxsize])
    positions = einops.rearrange(positions, 'zyx d h w -> d h w zyx')

    print('calculating distance')
    difference = np.abs(positions - center)  # (100, 100, 100, 3)
    distance = np.sum(difference ** 2, axis=-1) ** 0.5

    # calculate whether each pixel is inside or outside the circle
    print('calculating which pixels are in sphere')
    idx = distance < sphere_radius
    mask[idx] = 1

    distance_from_edge = distance_transform_edt(mask == 0)
    boundary_pixels = (distance_from_edge <= soft_edge_size) & (distance_from_edge != 0)
    normalised_distance_from_edge = (distance_from_edge[boundary_pixels] / soft_edge_size) * np.pi

    mask[boundary_pixels] = (0.5 * np.cos(normalised_distance_from_edge) + 0.5)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(mask)
    napari.run()

