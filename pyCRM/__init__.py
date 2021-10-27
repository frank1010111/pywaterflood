"pyCRM"
from importlib.metadata import version

__version__ = version("pyCRM")
from .CRM import CRM
from .multiwellproductivity import calc_gains_homogeneous, translate_locations
