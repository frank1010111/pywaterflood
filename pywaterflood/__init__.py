"pywaterflood"
from importlib.metadata import version

__version__ = version("pywaterflood")
from .crm import CRM
from .multiwellproductivity import (  # noqa: F401
    calc_gains_homogeneous,
    translate_locations,
)

__all__ = ["CRM", "calc_gains_homogeneous", "translate_location"]
