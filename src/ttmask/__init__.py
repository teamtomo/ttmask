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
from .test import test

