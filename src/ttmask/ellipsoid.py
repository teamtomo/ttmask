"""Ellipsoid mask creation."""

from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np
import typer
from typing_extensions import Annotated

from ._cli import cli
from .box_setup import box_setup
from .soft_edge import add_soft_edge


def ellipsoid(
    sidelength: int,
    ellipsoid_dimensions: Tuple[float, float, float],
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple,
) -> np.ndarray:
    """Create an ellipsoid mask.

    Creates a 3D mask in the shape of an ellipsoid.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    ellipsoid_dimensions : Tuple[float, float, float]
        The dimensions (z, y, x) of the ellipsoid in physical units
    wall_thickness : float
        The thickness of the ellipsoid wall in physical units
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
        3D numpy array containing the ellipsoid mask
    """
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)

    # converting relative coordinates to xyz distances (i.e. not a negative number) :
    xyz_distances = np.abs(coordinates_centered)

    # extract xyz magnitudes from xyz_distances
    x_magnitude = xyz_distances[:, :, :, 2]
    y_magnitude = xyz_distances[:, :, :, 1]
    z_magnitude = xyz_distances[:, :, :, 0]

    # define desired dimensions in pixels (converting from angstrom)
    z_axis_length = ellipsoid_dimensions[0] / (2 * pixel_size)
    y_axis_length = ellipsoid_dimensions[1] / (2 * pixel_size)
    x_axis_length = ellipsoid_dimensions[2] / (2 * pixel_size)

    # set up criteria for which pixels are inside the ellipsoid and modify values to 1.
    in_ellipsoid = (((x_magnitude) ** 2) / (x_axis_length**2)) + (
        (y_magnitude**2) / (y_axis_length**2)
    ) + ((z_magnitude**2) / (z_axis_length**2)) <= 1
    mask[in_ellipsoid] = 1

    # if requested, criteria set up for pixels within the hollowed area
    # and these values changed to zero
    if wall_thickness != 0:
        in_hollowing = (
            ((x_magnitude) ** 2) / ((x_axis_length - wall_thickness) ** 2)
        ) + ((y_magnitude**2) / ((y_axis_length - wall_thickness) ** 2)) + (
            (z_magnitude**2) / ((z_axis_length - wall_thickness) ** 2)
        ) <= 1
        mask[in_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask


@cli.command(name="ellipsoid")  # type: ignore[misc]
def ellipsoid_cli(
    sidelength: int = typer.Option(...),
    ellipsoid_dimensions: Annotated[Tuple[float, float, float], typer.Option()] = (
        0.0,
        0.0,
        0.0,
    ),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("ellipsoid.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50),
) -> None:
    """Create an ellipsoid mask.

    Creates a 3D mask in the shape of an ellipsoid.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    ellipsoid_dimensions : Tuple[float, float, float]
        The dimensions (z, y, x) of the ellipsoid in physical units
    wall_thickness : float
        The thickness of the ellipsoid wall in physical units
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
    mask = ellipsoid(
        sidelength,
        ellipsoid_dimensions,
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
