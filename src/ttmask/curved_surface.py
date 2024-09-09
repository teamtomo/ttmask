from pathlib import Path
import numpy as np
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup

def curved_surface(sidelength: int, fit_sphere_diameter: float, soft_edge_width: int, pixel_size: float) -> np.ndarray:
    fit_sphere_radius = fit_sphere_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)

    # determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_centered, axis=-1)

    # set up criteria for which pixels are inside the curved surface and modify values to 1.
    inside_curved_surface = distance_to_center < (fit_sphere_radius / pixel_size)
    mask[inside_curved_surface] = 1

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='curved_surface')
def curved_surface_cli(
    sidelength: int = typer.Option(...),
    fit_sphere_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("curved_surface.mrc")),
):
    mask = curved_surface(sidelength, fit_sphere_diameter, soft_edge_width, pixel_size)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
