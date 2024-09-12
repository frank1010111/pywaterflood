"""Functions for estimating aquifer influx using van Everdingen-Hurst models.

Through some accident of nature, oil tends to migrate through aquifers from where it was generated
to traps that eventually become the reservoirs we know and love. This means that almost all
reservoirs have associated aquifers. As oil is produced from these reservoirs, the aquifer water
enters the reservoir, providing pressure support and sometimes being co-produced. van Everdingen
and Hurst set out to model the reservoir-aquifer interaction in the 1940s. At the time, their
calculations were provided to petroleum engineers in the form of tables, but with modern computing,
we can improve upon that.


Further reading
---------------

`Petrowiki on water influx models <https://petrowiki.spe.org/Water_influx_models>`_

van Everdingen and Hurst, 1949, https://doi.org/10.2118/949305-G

Klins et al, 1988, http://dx.doi.org/10.2118/15433-PA
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.special import j0, j1, y0, y1


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


def aquifer_production(
    delta_pressure: NDArray,
    t_d: NDArray,
    r_ed: float,
    method: Literal["marsal-walsh", "klins"] = "klins",
    aquifer_constant: float = 1.0,
) -> NDArray:
    r"""Calculate cumulative aquifer production.

    Parameters
    ----------
    delta_pressure : NDArray
        change in reservoir pressure
    t_d : NDArray
        dimensionless time
    r_ed : float
        dimensionless radius, :math:`r_a/r_o`
    method : Literal str
        choose 'marsal-walsh' or 'klins' approximation
    aquifer_constant : float
        aquifer constant in RB/psi

        :math:`1.119 hfr^2_o \phi c_t`

    Returns
    -------
    NDArray
        Cumulative production from the aquifer
    """
    n_t = len(t_d)
    w_ek = np.zeros(n_t)
    for k in range(1, n_t):
        t_dk = t_d[k] - np.array(t_d)[:k]
        w_ek[k] = sum(water_dimensionless(t_dk, r_ed, method) * delta_pressure[1 : k + 1])
    return aquifer_constant * w_ek


def water_dimensionless(
    time_d: float | NDArray[np.float64],
    r_ed: float,
    method: Literal["marsal-walsh", "klins"] = "klins",
) -> NDArray[np.float64]:
    r"""Calculate the radial aquifer dimensionless water influx.

    This acts as if the aquifer were infinite-acting if :math:`\max t_d < 0.4 (r_{ed}^2 - 1)`

    Parameters
    ----------
    time_D : float | NDArray[np.float64]
        dimensionless time
    r_ed : float
        dimensionless radius, :math:`r_a/r_o`
    method : Literal str
        choose 'marsal-walsh' or 'klins' approximation

    Returns
    -------
    NDArray[np.float64]
    """
    time_d = np.asarray(time_d)

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


def get_bessel_roots(
    r_ed: float, n_max: int, root_choice: Literal["alpha", "beta"] = "beta"
) -> NDArray[np.float64]:
    r"""Find roots of Bessel function in Klins.

    Comes from Klins, 1988, eq 9,

    .. math::

        J_1(\beta_n r_{eD}) Y_1(\beta_n) - J_1(\beta_n) Y_1(\beta_n r_{eD})

    and eq 16

    .. math::
        J_1(\alpha_n r_{eD}) Y_0(\alpha_n) - Y_1(\alpha_n) J_0(\alpha_n r_{eD})

    Parameters
    ----------
    r_ed : float
        dimensionless reservoir radius, :math:`r_o/r_a`
    n_max : int
        number of roots to find
    root_choice: str
        which roots to get.

        for eq 9, `beta`, for eq 16, `alpha`

    Returns
    -------
    numpy array:
        roots of the function

    Reference
    ---------
    Klins, M. A., Bouchard, A. J., and C. L. Cable.
    "A Polynomial Approach to the van Everdingen-Hurst Dimensionless Variables for
    Water Encroachment." SPE Res Eng 3 (1988): 320-326.
    doi: <https://doi.org/10.2118/15433-PA>
    """

    def root_func_beta(beta):
        return j1(beta * r_ed) * y1(beta) - j1(beta) * y1(beta * r_ed)

    def root_func_alpha(alpha):
        return j1(alpha * r_ed) * y0(alpha) - y1(alpha * r_ed) * j0(alpha)

    if root_choice == "beta":
        root_func = root_func_beta
    elif root_choice == "alpha":
        root_func = root_func_alpha
    else:
        msg = f"root choice must be either alpha or beta, but {root_choice}"
        raise ValueError(msg)

    sample_roots = np.linspace(1e-9, 8 * n_max / r_ed, n_max * 400)
    zero_crossings = np.array([])
    while len(zero_crossings) < n_max:
        sample_roots *= 2
        zero_crossings = np.where(np.diff(np.sign(root_func(sample_roots))))[0]
    zero_crossings = zero_crossings[:n_max]

    roots = [root_scalar(root_func, x0=sample_roots[zc]).root for zc in zero_crossings]
    return np.asarray(roots)


def klins_pressure_dimensionless(
    t_d: NDArray[np.float64], r_ed: float, max_terms: int = 20
) -> NDArray[np.float64]:
    """Dimensionless pressure for a finite aquifer over time.

    Parameters
    ----------
    t_d : numpy array
        dimensionless time
    r_ed : float
        dimensionless radius
    max_terms : int, defaults to 20
        Number of terms in the infinite series to include. Klins used 2.

    Returns
    -------
    NDArray :
        dimensionless pressure
    """
    first_term = 2 / (r_ed**2 - 1) * (0.25 + t_d)
    second_term = -(3 * r_ed**4 - 4 * r_ed**4 * np.log(r_ed) - 2 * r_ed**2 - 1) / (
        4 * (r_ed**2 - 1) ** 2
    )
    third_term = 0.0
    betas = get_bessel_roots(r_ed, max_terms, "beta")
    for n in range(max_terms):
        third_term += (
            2
            * (np.exp(-(betas[n] ** 2) * t_d) * j1(betas[n] * r_ed) ** 2)
            / (betas[n] ** 2 * (j1(betas[n] * r_ed) ** 2 - j1(betas[n]) ** 2))
        )
    return first_term + second_term + third_term


def klins_water_dimensionless_finite(
    t_d: NDArray[np.float64], r_ed: float, max_terms: int = 20
) -> NDArray[np.float64]:
    """Solve for water influx from Klins' finite radial reservoir.

    Parameters
    ----------
    t_d : NDArray[np.float64]
        dimensionless time
    r_ed : float
        dimensionless radius
    max_terms : int, optional
        Number of terms in the infinite series to include when approximating the solution,
        by default 20

    Returns
    -------
    NDArray[np.float64]
        :math:`W_d`, dimensionless water influx
    """
    first_term = 0.5 * (r_ed**2 - 1)
    second_term = np.zeros_like(t_d)
    alphas = get_bessel_roots(r_ed, max_terms, "alpha")
    for n in range(max_terms):
        second_term += (
            -2
            * np.exp(-(alphas[n] ** 2) * t_d)
            * (j1(alphas[n] * r_ed) ** 2)
            / (alphas[n] ** 2 * (j0(alphas[n]) ** 2 - j1(alphas[n] * r_ed) ** 2))
        )
    return first_term + second_term
