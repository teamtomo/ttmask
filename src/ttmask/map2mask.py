import mrcfile
import numpy as np
import typer
from pathlib import Path

from ._cli import cli
from .soft_edge import add_soft_edge
from .add_padding import add_padding

@cli.command(name='map2mask')
def map2mask(

    input_map: Path = typer.Option(Path("map.mrc")),
    binarization_threshold: float = typer.Option(...),
    output_mask: Path = typer.Option(Path("mask.mrc")),
    pixel_size: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    padding_width: int = typer.Option(0),
):
    with mrcfile.open(input_map) as mrc:
        map_data = np.array(mrc.data)

    above_threshold = map_data >= binarization_threshold
    below_threshold = map_data < binarization_threshold

    map_data[above_threshold] = 1
    map_data[below_threshold] = 0

    padded_mask = add_padding(map_data, padding_width)
    mask = add_soft_edge(padded_mask, soft_edge_width)

    mrcfile.write(output_mask, mask, voxel_size=pixel_size, overwrite=True)