"""Cone mask creation."""

from pathlib import Path
from typing import Tuple

import einops
import mrcfile
import numpy as np
import typer
from typing_extensions import Annotated

from ._cli import cli
from .box_setup import box_setup
from .soft_edge import add_soft_edge


def cone(
    sidelength: int,
    cone_height: float,
    cone_base_diameter: float,
    soft_edge_width: int,
    pixel_size: float,
    centering: str,
    center: tuple,
) -> np.ndarray:
    """Create a cone mask.

    Creates a 3D mask in the shape of a cone.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    cone_height : float
        The height of the cone in physical units
    cone_base_diameter : float
        The diameter of the cone base in physical units
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
        3D numpy array containing the cone mask
    """
    # establish our coordinate system and empty mask
    coordinates_centered, mask = box_setup(sidelength, centering, center)
    # distances between each pixel and center :
    magnitudes = np.linalg.norm(coordinates_centered, axis=-1)
    magnitudes = einops.rearrange(magnitudes, "d h w -> d h w 1")

    # Check for zeros in magnitudes, replace them with eps to avoid Nan warning
    near_zero = 1e-8
    magnitudes = np.where(magnitudes == 0, near_zero, magnitudes)
    normalised = coordinates_centered / magnitudes

    principal_axis = np.array([1, 0, 0])
    dot_product = np.dot(normalised, principal_axis)
    angles_radians = np.arccos(dot_product)
    angles = np.rad2deg(angles_radians)

    z_distance = coordinates_centered[:, :, :, 0]  # (100, 100, 100)

    # Calculate the angle from the tip of the cone to the edge of the base
    cone_base_radius = (cone_base_diameter / 2) / pixel_size
    cone_angle = np.rad2deg(np.arctan(cone_base_radius / cone_height))

    within_cone_height = z_distance < (cone_height / pixel_size)
    within_cone_angle = angles < cone_angle

    # mask[within_cone_height] = 1
    mask[np.logical_and(within_cone_height, within_cone_angle)] = 1

    # need to adjust the center of the hollow cone
    # otherwise the cone thins towards the apex
    # thickness will need to decrease towards the apex anyway -
    # it's surely not possible / realistic?

    # Shift the mask in the z-axis by cone_height / 2
    z_shift = -int(cone_height / 2)
    mask = np.roll(mask, z_shift, axis=0)
    mask = add_soft_edge(mask, soft_edge_width)

    return mask


@cli.command(name="cone")  # type: ignore[misc]
def cone_cli(
    sidelength: int = typer.Option(...),
    cone_height: float = typer.Option(...),
    cone_base_diameter: float = typer.Option(...),
    soft_edge_width: int = typer.Option(0),
    pixel_size: float = typer.Option(1),
    output: Path = typer.Option(Path("cone.mrc")),
    centering: str = typer.Option("standard"),
    center: Annotated[Tuple[int, int, int], typer.Option()] = (50, 50, 50),
) -> None:
    """Create a cone mask.

    Creates a 3D mask in the shape of a cone.

    Parameters
    ----------
    sidelength : int
        The sidelength of the cubic mask volume in pixels
    cone_height : float
        The height of the cone in physical units
    cone_base_diameter : float
        The diameter of the cone base in physical units
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
    mask = cone(
        sidelength,
        cone_height,
        cone_base_diameter,
        soft_edge_width,
        pixel_size,
        centering,
        center,
    )

    # Save the mask to an MRC file
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = pixel_size
