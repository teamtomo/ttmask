"""Sphere mask creation."""

from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np
import typer
from typing_extensions import Annotated

from ._cli import cli
from .box_setup import box_setup
from .soft_edge import add_soft_edge


def sphere(
    sidelength: int,
    sphere_diameter: float,
    wall_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple,
) -> np.ndarray:
    """Create a sphere mask.

    Creates a 3D mask in the shape of a sphere.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    sphere_diameter : float
        The diameter of the sphere in physical units
    wall_thickness : float
        The thickness of the sphere wall in physical units
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
        3D numpy array containing the sphere mask
    """
    sphere_radius = sphere_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)

    # determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_centered, axis=-1)

    # set up criteria for which pixels are inside the sphere and modify values to 1.
    inside_sphere = distance_to_center < (sphere_radius / pixel_size)
    mask[inside_sphere] = 1

    # if requested, criteria set up for pixels within the hollowed area
    # and these values changed to zero
    if wall_thickness != 0:
        within_hollowing = distance_to_center < (
            (sphere_radius - wall_thickness) / pixel_size
        )
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask


@cli.command(name="sphere")  # type: ignore[misc]
def sphere_cli(
    sidelength: int = typer.Option(...),
    sphere_diameter: float = typer.Option(...),
    wall_thickness: float = typer.Option(0),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("sphere.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50),
) -> None:
    """Create a sphere mask.

    Creates a 3D mask in the shape of a sphere.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    sphere_diameter : float
        The diameter of the sphere in physical units
    wall_thickness : float
        The thickness of the sphere wall in physical units
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
    mask = sphere(
        sidelength,
        sphere_diameter,
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
