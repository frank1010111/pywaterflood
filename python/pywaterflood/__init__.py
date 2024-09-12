"""``pywaterflood`` A package for explaining and predicting waterflood performance.

Pywaterflood is focused on numerical modeling of oil reservoirs in secondary or
tertiary recovery. These include methods for capacitance resistance modeling,
the multiwell productivity index, Buckley-Leverett displacement, and aquifer drive.

Provides
========

``CRM`` a class for performing capacitance resistance modeling

``CrmCompensated`` a class for performing pressure-compensated CRM

``calc_gains_homogeneous`` a function using the multiwell productivity index

``translate_location`` a function for moving things from the real world into dimensionless-land
For more information on these functions and more, check the submodule API documentation.
"""

from __future__ import annotations

import sys

from pywaterflood.crm import CRM, CrmCompensated
from pywaterflood.multiwellproductivity import (
    calc_gains_homogeneous,
    translate_locations,
)

pyversion = sys.version_info
if pyversion.major == 3 and pyversion.minor > 7:
    from importlib.metadata import version

    __version__ = version("pywaterflood")
del sys, pyversion

__all__ = ["CRM", "CrmCompensated", "calc_gains_homogeneous", "translate_locations"]
