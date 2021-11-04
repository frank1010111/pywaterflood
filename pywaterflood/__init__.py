"pywaterflood"
from importlib.metadata import version

__version__ = version("pywaterflood")
from .crm import CRM
from .multiwellproductivity import calc_gains_homogeneous, translate_locations
