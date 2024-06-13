"""CLI tool for mask creation in cryo-EM/ET"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttmask")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Miles Graham"
__email__ = "miles.graham@balliol.ox.ac.uk"

from ._cli import cli
from .sphere import sphere
from .cylinder import cylinder
from .cuboid import cuboid
from .cube import cube
from .cone import cone
from .ellipsoid import ellipsoid
from .map2mask import map2mask
from .tube import tube

