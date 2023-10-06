from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def effective_reservoir_radius(
    reservoir_pore_volume: float, porosity: float, thickness: float, flow_solid_angle: float = 360
) -> float:
    """Calculate effective reservoir_radius

    Parameters
    ----------
    reservoir_pore_volume : float
        water pore volume in reservoir bbl
    porosity : float
        porosity fraction, ranges from 0-1
    thickness : float
        pay thickness in feet
    flow_solid_angle : float
        solid angle between well and reservoir in degrees, ranges from 0-360

    Returns
    -------
    float
        effective reservoir radius, $r_o$, in feet
    """
    if porosity > 1:
        msg = f"porosity can't be above 1, it's {porosity}"
        raise ValueError(msg)
    if flow_solid_angle > 360:
        msg = f"Flow angle can't exceed 360 (a full circle), but it's {flow_solid_angle}"
        raise ValueError(msg)
    return math.sqrt(
        5.6146 * reservoir_pore_volume / (math.pi * porosity * thickness * flow_solid_angle / 360)
    )


def water_dimensionless_infinite(time_D: float) -> float:
    """Calculate infinite acting radial aquifer dimensionless water influx.

    Parameters
    ----------
    time_D : float
        dimensionless time

    Returns
    -------
    float
        dimensionless water influx
    """
    return np.cumsum(
        2 * np.sqrt(time_D / math.pi)
        + time_D / 2
        - time_D / 6 * np.sqrt(time_D / math.pi)
        + time_D**2 / 16
    )


def aquifer_production_infinite(
    delta_pressure: NDArray, t_D: NDArray, aquifer_constant: float
) -> NDArray:
    """Calculate cumulative aquifer production assuming infinite boundaries.

    Parameters
    ----------
    delta_pressure : NDArray
        change in reservoir pressure
    t_D : NDArray
        dimensionless time
    aquifer_constant : float
        aquifer constant in RB/psi

        $1.119 hfr^2_o /phi c_t$

    Returns
    -------
    NDArray
        Cumulative production from the aquifer
    """
    t_D_diff = np.diff(t_D)
    water_influx_dimensionless = np.cumsum(water_dimensionless_infinite(t_D_diff))
    return aquifer_constant * delta_pressure * water_influx_dimensionless
