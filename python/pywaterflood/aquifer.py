from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pywaterflood import _core


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
        msg = f"flow_solid_angle can't exceed 360 (a full circle), but it's {flow_solid_angle}"
        raise ValueError(msg)
    return math.sqrt(
        5.6146 * reservoir_pore_volume / (math.pi * porosity * thickness * flow_solid_angle / 360)
    )


def water_dimensionless(
    time_d: float | NDArray[np.float64],
    r_ed: float,
    method: Literal["marsal-walsh", "klins"] = "marsal-walsh",
) -> NDArray[np.float64]:
    """Calculate the radial aquifer dimensionless water influx.

    Parameters
    ----------
    time_D : float | NDArray[np.float64]
        dimensionless time
    r_ed : float
        dimensionless radius, :math: r_a/r_o
    method : Literal str
        choose 'marsal-walsh' or 'klins' approximation

    Returns
    -------
    NDArray[np.float64]
    """
    if isinstance(time_d, float):
        time_d = np.array([time_d])

    if r_ed <= 1:
        msg = f"r_ed = r_aquifer/r_reservoir must be greater than 1, is {r_ed=}"
        raise ValueError(msg)

    water_influx_infinite = water_dimensionless_infinite(time_d, method)
    j_star = r_ed**4 * np.log(r_ed) / (r_ed**2 - 1) + 0.25 * (1 - 3 * r_ed**2)
    water_influx_finite = 0.5 * (r_ed**2 - 1) * (1 - np.exp(-2 * time_d / j_star))
    time_d_star = 0.4 * (r_ed**2 - 1)
    return np.where(time_d < time_d_star, water_influx_infinite, water_influx_finite)


def water_dimensionless_infinite(
    time_D: float | NDArray[np.float64],
    method: Literal["marsal-walsh", "klins"] = "marsal-walsh",
) -> NDArray[np.float64]:
    """Calculate infinite acting radial aquifer dimensionless water influx.

    Parameters
    ----------
    time_D : float | NDArray[np.float64]
        dimensionless time

    Returns
    -------
    NDArray[np.float64]
        dimensionless water influx
    """
    time_D = np.asarray(time_D)

    water_influx = np.zeros_like(time_D)
    if method == "marsal-walsh":
        # short time
        short_time = time_D <= 1
        water_influx[short_time] = (
            2 * np.sqrt(time_D[short_time] / math.pi)
            + time_D[short_time] / 2
            - time_D[short_time] / 6 * np.sqrt(time_D[short_time] / math.pi)
            + time_D[short_time] ** 2 / 16
        )
        # medium time
        med_time = (time_D > 1) & (time_D <= 100)
        a_coefficients = [
            8.1638e-1,
            8.5373e-1,
            -2.7455e-2,
            1.0284e-3,
            -2.274e-5,
            2.8354e-7,
            -1.8436e-9,
            4.8534e-12,
        ]
        water_influx[med_time] = sum(a_coefficients[i] * time_D[med_time] ** i for i in range(8))
        # long time
        long_time = time_D > 100
        water_influx[long_time] = 2 * time_D[long_time] / np.log(time_D[long_time])
    elif method == "klins":
        # short time
        short_time = time_D <= 0.01
        water_influx[short_time] = np.sqrt(time_D[short_time] / np.pi)
        # medium time
        med_time = (time_D > 0.01) & (time_D <= 200)
        sqrt_time = np.sqrt(time_D[med_time])
        water_influx[med_time] = (
            1.2838 * sqrt_time
            + 1.19328 * time_D[med_time]
            + 0.269872 * sqrt_time**3
            + 0.00855294 * time_D[med_time] ** 2
        ) / (1 + 0.616599 * sqrt_time + 0.0413008 * time_D[med_time])
        # long time
        long_time = time_D > 200
        water_influx[long_time] = (-4.29881 + 2.02566 * time_D[long_time]) / np.log(
            time_D[long_time]
        )
    else:
        msg = f"method must be 'marsal-walsh' or 'klins', but '{method}' was given"
        raise ValueError(msg)
    return water_influx


def aquifer_production(delta_pressure: NDArray, w_d: NDArray, aquifer_constant: float) -> NDArray:
    """Calculate cumulative aquifer production.

    Parameters
    ----------
    delta_pressure : NDArray
        change in reservoir pressure
    w_d : NDArray
        dimensionless water influx
    aquifer_constant : float
        aquifer constant in RB/psi

        $1.119 hfr^2_o /phi c_t$

    Returns
    -------
    NDArray
        Cumulative production from the aquifer
    """
    w_ek = _core.w_ek(w_d, delta_pressure)
    return aquifer_constant * w_ek
