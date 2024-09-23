from pathlib import Path
import numpy as np
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup

def curved_surface(
    sidelength: int, 
    fit_sphere_diameter: float,
    surface_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str
) -> np.ndarray:
    sphere_radius = fit_sphere_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering)
    coordinates_shifted = coordinates_centered - ([0, sphere_radius, 0])


    #determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_shifted, axis=-1)


    # set up criteria for which pixels are inside the sphere and modify values to 1.
    inside_sphere = distance_to_center < (sphere_radius / pixel_size)
    mask[inside_sphere] = 1

    # if requested, criteria set up for pixels within the hollowed area and these values changed to zero
    if surface_thickness != 0:
        within_hollowing = distance_to_center < ((sphere_radius - surface_thickness) / pixel_size)
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='curved_surface')
def curved_surface_cli(
    sidelength: int = typer.Option(...),
    fit_sphere_diameter: float = typer.Option(...),
    surface_thickness: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("curved_surface.mrc")),
    centering: str = typer.Option("standard"),
):
    mask = curved_surface(sidelength, fit_sphere_diameter, surface_thickness, soft_edge_width, pixel_size, centering)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
