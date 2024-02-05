"""Estimations for behavior of waterflood fronts from Buckley and Leverett's theory."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pywaterflood import _core


@dataclass
class Reservoir:
    phi: float
    """effective porosity"""
    viscosity_oil: float
    """oil viscosity in Pa⋅s"""
    viscosity_water: float
    """water viscosity in Pa⋅s"""
    sat_oil_r: float
    """residual oil saturation"""
    sat_water_c: float
    """critical (residual) water saturation"""
    sat_gas_c: float
    """critical gas saturation"""
    n_oil: float
    """Brooks-Corey exponent for oil rel-perm"""
    n_water: float
    """Brooks-Corey exponent for water rel-perm"""
    flow_cross_section: float = 1.0
    """Area flow is through in m$^2$, defaults to 1"""

    def __post_init__(self):
        """Validate inputs after initialization."""
        for key in [
            "phi",
            "flow_cross_section",
            "viscosity_oil",
            "viscosity_water",
            "sat_oil_r",
            "sat_water_c",
            "sat_gas_c",
        ]:
            value = getattr(self, key)
            if value < 0:
                msg = f"{key} must be greater than zero, it is {value}"
                raise ValueError(msg)
        for key in ["phi", "sat_oil_r", "sat_water_c", "sat_gas_c"]:
            value = getattr(self, key)
            if value > 1:
                msg = f"{key} must be between 0 and one. It is set to {value}"
                raise ValueError(msg)
        for key in ["n_oil", "n_water"]:
            value = getattr(self, key)
            if (value < 1) or (value > 6):
                msg = f"{key} must be between 1 and 6. It is {value}"
                raise ValueError(msg)
        summed_residual_sats = self.sat_water_c + self.sat_oil_r + self.sat_gas_c
        if summed_residual_sats > 1.0:
            msg = f"Sum of the residual saturations cannot exceed one {summed_residual_sats=}"
            raise ValueError(msg)


def water_front_velocity(reservoir: Reservoir, sat_water: float, flow_rate: float):
    r"""Calculate the velocity for the water front at a particular water saturation.

    Above the breakthrough saturation follows this equation:

    .. math::
        \left(\frac{dx}{dt}\right)_{S_w}
        = \frac{q_t}{\phi A} \left(\frac{\partial f_w}{\partial S_w}\right)_t

    Below the breakthrough saturation, the velocity is equal to the velocity for the equation
    at breakthrough saturation.

    Parameters
    ----------
    reservoir: Reservoir
        The reservoir parameters between the injection and production wells.
    sat_water: float
        fraction of fluid that is water
    flow_rate: float
        water injection rate in m$^3$/d

    Returns
    -------
    the front velocity in m/d
    """
    # assume no free gas
    sat_oil = 1 - sat_water
    return _core.water_front_velocity(
        sat_water=sat_water, sat_oil=sat_oil, flow_rate=flow_rate, **asdict(reservoir)
    )


def breakthrough_sw(reservoir: Reservoir):
    """Calculate the water saturation at the front at breakthrough.

    Parameters
    ----------
    reservoir : Reservoir
        Parameters of the reservoir rock and fluid between the injector and producer
    """
    params = {
        key: value
        for key, value in asdict(reservoir).items()
        if key
        in [
            "viscosity_oil",
            "viscosity_water",
            "sat_oil_r",
            "sat_water_c",
            "sat_gas_c",
            "n_oil",
            "n_water",
        ]
    }
    return _core.breakthrough_sw(**params)
