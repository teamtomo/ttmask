"""Curved surface mask creation."""

from pathlib import Path

import mrcfile
import numpy as np
import typer

from ._cli import cli
from .box_setup import box_setup
from .soft_edge import add_soft_edge


def curved_surface(
    sidelength: int,
    fit_sphere_diameter: float,
    surface_thickness: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
) -> np.ndarray:
    """Create a curved surface mask.

    Creates a 3D mask in the shape of a curved surface (part of a sphere)
    with specified dimensions and properties.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    fit_sphere_diameter : float
        The diameter of the sphere used to create the curved surface in physical units
    surface_thickness : float
        The thickness of the curved surface in physical units
    soft_edge_width : int
        Width of the soft edge in pixels
    pixel_size : float
        Physical size of each pixel
    centering : str
        Method for centering the mask in the volume

    Returns
    -------
    np.ndarray
        3D numpy array containing the curved surface mask
    """
    sphere_radius = fit_sphere_diameter / 2

    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering)
    coordinates_shifted = coordinates_centered - ([0, sphere_radius, 0])

    # determine distances of each pixel to the center
    distance_to_center = np.linalg.norm(coordinates_shifted, axis=-1)

    # set up criteria for which pixels are inside the sphere and modify values to 1.
    inside_sphere = distance_to_center < (sphere_radius / pixel_size)
    mask[inside_sphere] = 1

    # if requested, criteria set up for pixels within the hollowed area
    # and these values changed to zero
    if surface_thickness != 0:
        within_hollowing = distance_to_center < (
            (sphere_radius - surface_thickness) / pixel_size
        )
        mask[within_hollowing] = 0

    # if requested, a soft edge is added to the mask
    mask = add_soft_edge(mask, soft_edge_width)

    return mask


@cli.command(name="curved_surface")  # type: ignore[misc]
def curved_surface_cli(
    sidelength: int = typer.Option(...),
    fit_sphere_diameter: float = typer.Option(...),
    surface_thickness: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("curved_surface.mrc")),
    centering: str = typer.Option("standard"),
) -> None:
    """Create a curved surface mask.

    Creates a 3D mask in the shape of a curved surface (part of a sphere)
    with specified dimensions and properties.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    fit_sphere_diameter : float
        The diameter of the sphere used to create the curved surface in physical units
    surface_thickness : float
        The thickness of the curved surface in physical units
    soft_edge_width : int
        Width of the soft edge in pixels
    pixel_size : float
        Physical size of each pixel
    output : Path
        Path to save the output MRC file
    centering : str
        Method for centering the mask in the volume
    """
    mask = curved_surface(
        sidelength,
        fit_sphere_diameter,
        surface_thickness,
        soft_edge_width,
        pixel_size,
        centering,
    )

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
