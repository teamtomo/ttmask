from pathlib import Path
import numpy as np
import typer
import mrcfile

from .soft_edge import add_soft_edge
from ._cli import cli
from .box_setup import box_setup

def cube(sidelength: int, cube_sidelength: float, soft_edge_width: float, pixel_size: float) -> np.ndarray:
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength)

    # determine distances of each pixel to the center
    half_cube_side = cube_sidelength / 2 / pixel_size
    inside_cube = np.all(np.abs(coordinates_centered) <= half_cube_side, axis=-1)
    mask[inside_cube] = 1

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask

@cli.command(name='cube')
def cube_cli(
    sidelength: int = typer.Option(...),
    cube_sidelength: float = typer.Option(...),
    soft_edge_width: float = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cube.mrc")),
):
    mask = cube(sidelength, cube_sidelength, soft_edge_width, pixel_size)

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
