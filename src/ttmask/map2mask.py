"""Convert 3D volumetric maps to binary masks."""

from pathlib import Path

import mrcfile
import numpy as np
import typer

from ._cli import cli
from .add_padding import add_padding
from .soft_edge import add_soft_edge


def mask_from_map(
    map_data: np.ndarray,
    binarization_threshold: float,
    padding_width: int,
    soft_edge_width: int,
) -> np.ndarray:
    """Create a binary mask from a 3D map.

    Binarizes map using a threshold value, adds optional padding, applies a soft edge.

    Parameters
    ----------
    map_data : np.ndarray
        The input 3D map data
    binarization_threshold : float
        Threshold value for binarization (values above become 1, below become 0)
    padding_width : int
        Width of padding to add around the mask in pixels
    soft_edge_width : int
        Width of the soft edge in pixels

    Returns
    -------
    np.ndarray
        3D numpy array containing the binary mask
    """
    # Create a copy of the input data to make it writable
    map_data = map_data.copy()

    above_threshold = map_data >= binarization_threshold
    below_threshold = map_data < binarization_threshold

    map_data[above_threshold] = 1
    map_data[below_threshold] = 0

    padded_mask = add_padding(map_data, padding_width)
    mask = add_soft_edge(padded_mask, soft_edge_width)

    return mask


@cli.command(name="map2mask")  # type: ignore[misc]
def map2mask(
    input_map: Path = typer.Option(Path("map.mrc")),
    binarization_threshold: float = typer.Option(...),
    padding_width: int = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(...),
    output_mask: Path = typer.Option(Path("mask.mrc")),
) -> None:
    """Convert a 3D map to a binary mask.

    Loads a 3D map from an MRC file, binarizes it using a threshold value,
    adds optional padding and soft edge, and saves the resulting mask.

    Parameters
    ----------
    input_map : Path
        Path to the input map MRC file
    binarization_threshold : float
        Threshold value for binarization (values above become 1, below become 0)
    padding_width : int
        Width of padding to add around the mask in pixels
    soft_edge_width : int
        Width of the soft edge in pixels
    pixel_size : float
        Physical size of each pixel to be saved in the output file
    output_mask : Path
        Path to save the output mask MRC file
    """
    with mrcfile.open(input_map, permissive=True) as mrc:
        data = mrc.data
    mask = mask_from_map(data, binarization_threshold, padding_width, soft_edge_width)

    with mrcfile.new(output_mask, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
