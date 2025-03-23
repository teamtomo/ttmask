"""Tube mask creation."""

from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np
import typer
from typing_extensions import Annotated

from ._cli import cli
from .box_setup import box_setup
from .soft_edge import add_soft_edge


def tube(
    sidelength: int,
    tube_height: float,
    tube_diameter: float,
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple,
) -> np.ndarray:
    """Create a tube mask.

    Creates a 3D mask in the shape of a tube.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    tube_height : float
        The height of the tube in physical units
    tube_diameter : float
        The diameter of the tube in physical units
    wall_thickness : float
        The thickness of the tube wall in physical units
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
        3D numpy array containing the tube mask
    """
    tube_radius = tube_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)

    # converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # set up criteria for which pixels are inside the tube and modify values to 1.
    xy_distance = np.linalg.norm(xyz_distances[:, :, :, [1, 2]], axis=-1)
    within_z = xyz_distances[:, :, :, 0] < (tube_height / (2 * pixel_size))
    within_xy = xy_distance < (tube_radius / pixel_size)
    mask[np.logical_and(within_z, within_xy)] = 1

    # if requested, criteria set up for pixels within the hollowed area
    # and these values changed to zero
    if wall_thickness != 0:
        within_xy_hollowing = xy_distance < (
            (tube_radius - wall_thickness) / pixel_size
        )
        mask[within_xy_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask


@cli.command(name="tube")  # type: ignore[misc]
def tube_cli(
    sidelength: int = typer.Option(...),
    tube_height: float = typer.Option(...),
    tube_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("tube.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50),
) -> None:
    """Create a tube mask.

    Creates a 3D mask in the shape of a tube.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    tube_height : float
        The height of the tube in physical units
    tube_diameter : float
        The diameter of the tube in physical units
    wall_thickness : float
        The thickness of the tube wall in physical units
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

    Returns
    -------
    np.ndarray
        3D numpy array containing the tube mask
    """
    mask = tube(
        sidelength,
        tube_height,
        tube_diameter,
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
