import mrcfile
import numpy as np
import typer
from pathlib import Path

from ._cli import cli
from .soft_edge import add_soft_edge
from .add_padding import add_padding

def mask_from_map(
    map_data: np.ndarray, 
    binarization_threshold: float,
    padding_width: int,
    soft_edge_width: int
) -> np.ndarray:

    above_threshold = map_data >= binarization_threshold
    below_threshold = map_data < binarization_threshold

    map_data[above_threshold] = 1
    map_data[below_threshold] = 0

    padded_mask = add_padding(map_data, padding_width)
    mask = add_soft_edge(padded_mask, soft_edge_width)

    return mask

@cli.command(name='map2mask')
def map2mask(
    input_map: Path = typer.Option(Path("map.mrc")),
    binarization_threshold: float = typer.Option(...),
    padding_width: int = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output_mask: Path = typer.Option(Path("mask.mrc")),
):
    with mrcfile.open(input_map, permissive=True) as mrc:
        data = mrc.data
    mask = mask_from_map(data, binarization_threshold, padding_width, soft_edge_width)

    with mrcfile.new(output_mask, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size