"""pywaterflood: A package for explaining and predicting waterflood performance.

Provides
--------
CRM : a class for performing capacitance resistance modeling
calc_gains_homogeneous : a function using the multiwell productivity index
translate_location : a function for moving things from the real world
  into dimensionless-land
"""
from __future__ import annotations

import sys

from pywaterflood.crm import CRM, CrmCompensated
from pywaterflood.multiwellproductivity import (  # noqa: F401
    calc_gains_homogeneous,
    translate_locations,
)

pyversion = sys.version_info
if pyversion.major == 3 and pyversion.minor > 7:
    from importlib.metadata import version

    __version__ = version("pywaterflood")
del sys, pyversion

__all__ = ["CRM", "CrmCompensated", "calc_gains_homogeneous", "translate_locations"]
