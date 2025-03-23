"""CLI tool for mask creation in cryo-EM/ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttmask")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Miles Graham"
__email__ = "miles.graham@balliol.ox.ac.uk"

from ._cli import cli
from .cone import cone
from .cube import cube
from .cuboid import cuboid
from .curved_surface import curved_surface
from .cylinder import cylinder
from .ellipsoid import ellipsoid
from .map2mask import map2mask
from .sphere import sphere
from .tube import tube

__all__ = [
    "cli",
    "cone",
    "cube",
    "cuboid",
    "curved_surface",
    "cylinder",
    "ellipsoid",
    "map2mask",
    "sphere",
    "tube",
]
