from pathlib import Path
import numpy as np
import einops
import typer
import mrcfile

from ._cli import cli
from .soft_edge import add_soft_edge
from .box_setup import box_setup

def cone(sidelength: int, cone_height: float, cone_base_diameter: float, soft_edge_width: int, pixel_size: float) -> np.ndarray:
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)

    # determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_centered[:, :, :, :2], axis=-1)
    height = coordinates_centered[:, :, :, 2]

    # set up criteria for which pixels are inside the cone and modify values to 1.
    inside_cone = (height >= 0) & (height <= (cone_height / pixel_size)) & (distance_to_center <= (cone_base_diameter / 2) * (1 - height / (cone_height / pixel_size)))
    mask[inside_cone] = 1

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='cone')
def cone_cli(
    sidelength: int = typer.Option(...),
    cone_height: float = typer.Option(...),
    cone_base_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cone.mrc")),
):
    mask = cone(sidelength, cone_height, cone_base_diameter, soft_edge_width, pixel_size)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
