"""Cuboid mask creation."""

from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np
import typer
from typing_extensions import Annotated

from ._cli import cli
from .box_setup import box_setup
from .soft_edge import add_soft_edge


def cuboid(
    sidelength: int,
    cuboid_sidelengths: Tuple[float, float, float],
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple,
) -> np.ndarray:
    """Create a cuboid mask.

    Creates a 3D mask in the shape of a cuboid.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    cuboid_sidelengths : Tuple[float, float, float]
        The sidelengths (x, y, z) of the cuboid in physical units
    wall_thickness : float
        The thickness of the cuboid wall in physical units
    soft_edge_width : int
        Width of the soft edge in pixels
    pixel_size : float
        Physical size of each pixel
    centering : str
        Method for centering the mask in the volume
    center : tuple
        (x,y,z) coordinates specifying the center position of the mask

    Returns
    -------
    np.ndarray
        3D numpy array containing the cuboid mask
    """
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)
    # converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # set up criteria for which pixels are inside the cuboid and modify values to 1.
    inside_cuboid = np.all(
        xyz_distances < (np.array(cuboid_sidelengths) / (2 * pixel_size)), axis=-1
    )
    mask[inside_cuboid] = 1

    # if requested, criteria set up for pixels within the hollowed area
    # and these values changed to zero
    if wall_thickness != 0:
        within_hollowing = np.all(
            xyz_distances
            < ((np.array(cuboid_sidelengths) / (2 * pixel_size)) - wall_thickness),
            axis=-1,
        )
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask


@cli.command(name="cuboid")  # type: ignore[misc]
def cuboid_cli(
    sidelength: int = typer.Option(...),
    cuboid_sidelengths: Annotated[Tuple[float, float, float], typer.Option()] = (
        0.0,
        0.0,
        0.0,
    ),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cuboid.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50),
) -> None:
    """Create a cuboid mask.

    Creates a 3D mask in the shape of a cuboid.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    cuboid_sidelengths : Tuple[float, float, float]
        The sidelengths (x, y, z) of the cuboid in physical units
    wall_thickness : float
        The thickness of the cuboid wall in physical units
    soft_edge_width : int
        Width of the soft edge in pixels
    pixel_size : float
        Physical size of each pixel
    output : Path
        Path to save the output MRC file
    centering : str
        Method for centering the mask in the volume
    center : tuple
        (x,y,z) coordinates specifying the center position of the mask
    """
    mask = cuboid(
        sidelength,
        cuboid_sidelengths,
        wall_thickness,
        soft_edge_width,
        pixel_size,
        centering,
        center,
    )

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
