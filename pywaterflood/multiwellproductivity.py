"""Interwell connectivity through geometric considerations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sl
from numba import njit
from numpy import ndarray

idx = pd.IndexSlice


def calc_gains_homogeneous(locations: pd.DataFrame, x_e: float, y_e: float) -> pd.DataFrame:
    """Calculate gains from injectors to producers using multiwell productivity index.

    Args
    ----------
    locations : pd.DataFrame
        columns include:
        `X` : x-location for well
        `Y` : y-location for well
        `Type`: ("Producer" or "Injector")
    x_e : float
        scaling in x-direction (unit size)
    y_e : float
        scaling in y-direction (unit size)

    Returns
    -------
    Lambda : pd.DataFrame
        contains the production-injection multiwell productivity index

    Notes
    -----
    This assumes a roughly rectangular unit with major axes at x and y. You might want
    to rotate your locations.

    References
    ----------
    Kaviani, D. and Valkó, P.P., 2010. Inferring interwell connectivity using \
    multiwell productivity index (MPI). Journal of Petroleum Science and Engineering, \
    73(1-2), p.48-58.
    """
    locations = locations.copy()
    locations[["X", "Y"]] /= x_e
    y_D = y_e / x_e
    A_prod = calc_influence_matrix(locations, y_D, "prod").astype(float)
    A_conn = calc_influence_matrix(locations, y_D, "conn").astype(float)
    A_prod_inv = sl.inv(A_prod.values)
    term1 = A_prod_inv / np.sum(A_prod_inv)
    term2 = np.ones_like(A_prod_inv) @ A_prod_inv @ A_conn.values - 1
    term3 = A_prod_inv @ A_conn.values
    Lambda = term1 @ term2 - term3
    connectivity_df = pd.DataFrame(Lambda, index=A_prod.index, columns=A_conn.columns).rename_axis(
        index="Producers", columns="Injectors"
    )
    return connectivity_df


def translate_locations(
    locations: pd.DataFrame, x_col: str, y_col: str, type_col: str
) -> pd.DataFrame:
    """Translate locations  to prepare for building connectivity matrix.

    Moves the lower left edge of the reservoir to (0, 0), and sets up the matrix columns
    to work with `calc_gains_homogeneous`

    Args
    ----------
    locations : pd.DataFrame
        Has the x,y locations and type assignment for all wells
    x_col : str
        Column in `locations` for the x-location
    y_col : str
        Column in `locations` for the y-location
    type_col : str
        Column in `locations` for the type (producer or injector)

    Returns
    -------
    locations_out: pd.DataFrame
        columns are `X` (x-location), `Y` (y-location), `Type` (producer or injector)
    """
    locations_out = pd.DataFrame(index=locations.index, columns=["X", "Y", "Type"])
    locations_out["X"] = locations[x_col] - locations[x_col].min()
    locations_out["Y"] = locations[y_col] - locations[y_col].min()
    locations_out["Type"] = locations[type_col]
    return locations_out


def calc_influence_matrix(
    locations: pd.DataFrame, y_D: float, matrix_type: str = "conn", m_max: int = 300
) -> pd.DataFrame:
    """Calculate influence matrix A.

    Args
    ----------
    locations : pd.DataFrame
        Has the x,y locations and type assignment for all wells
        columns are `X` (x-location), `Y` (y-location), `Type` (producer or injector)
    y_D : float
        dimensionless scaling for y-direction
    matrix_type : str, choice of `conn` or `prod`
        injector-producer matrix or producer-producer matrix
    m_max : int > 0
        number of terms in the series to calculate. 300 is a good default.

    Returns
    -------
    influence_matrix : pd.DataFrame
        a matrix with the influences between wells
    """
    assert matrix_type in ["conn", "prod"]
    XA = locations[locations.Type == "Producer"]
    if matrix_type == "prod":
        XB = XA.copy()
    else:
        XB = locations[locations.Type == "Injector"]
    influence_matrix = pd.DataFrame(
        index=pd.MultiIndex.from_product([XA.index, XB.index]), columns=["A"]
    )
    m = 1 + np.arange(m_max)  # elements of sum
    for i, j in influence_matrix.index:
        x_i, y_i = XA.loc[i, ["X", "Y"]]
        x_j, y_j = XB.loc[j, ["X", "Y"]] + 1e-6
        influence_matrix.loc[idx[i, j], "A"] = calc_A_ij(x_i, y_i, x_j, y_j, y_D, m)
    return influence_matrix["A"].unstack()


@njit
def calc_A_ij(x_i: float, y_i: float, x_j: float, y_j: float, y_D: float, m: ndarray) -> float:
    r"""Calculate element in the influence matrix.

    .. math::
        A_{ij} = 2 \pi y_D (\frac13 - \frac{y_i}{y_D} +
            \frac{y_i^2 + y_j^2}{2 y_D^2})
            + \sum_{m=1}^\infty \frac{t_m}m \cos(m\pi \tilde x_i)
            \cos(m \pi \tilde x_j)
    where

    .. math::
        t_m = \frac{\cosh\left(m\pi (y_D - |\tilde y_i - \tilde y_j|)\right)
        + \cosh\left(m\pi (y_D - \tilde y_i - \tilde y_j\right)}
        {\sinh\left(m\pi y_D \right)}

    Args
    ----
    x_i : float
        x-location of i'th well
    y_i : float
        y-location of i'th well
    x_j : float
        x-location of j'th well
    y_j : float
        y-location of j'th well
    y_D : float
        dimensionless parameter for y-direction
    m : ndarray
        series terms, from 1 to m_max

    Returns
    -------
    A_ij : float
    """
    first_term = 2 * np.pi * y_D * (1 / 3.0 - y_i / y_D + (y_i**2 + y_j**2) / (2 * y_D**2))
    return first_term + calc_summed_term(x_i, y_i, x_j, y_j, y_D, m)


@njit
def calc_summed_term(
    x_i: float, y_i: float, x_j: float, y_j: float, y_D: float, m: ndarray
) -> float:
    r"""Calculate summed term using Valkó 2000 equations A4-7.

    .. math::
        \sum_{m=1}^\infty \frac{t_m}m \cos(m\pi \tilde x_i)
        \cos(m \pi \tilde x_j)
    where

    .. math::
        t_m = \frac{\cosh\left(m\pi (y_D - |\tilde y_i - \tilde y_j|)\right)
        + \cosh\left(m\pi (y_D - \tilde y_i - \tilde y_j\right)}
        {\sinh\left(m\pi y_D \right)}

    Args
    ----
    x_i : float
        x-location of i'th well
    y_i : float
        y-location of i'th well
    x_j : float
        x-location of j'th well
    y_j : float
        y-location of j'th well
    y_D : float
        dimensionless parameter for y-direction
    m : ndarray
        series terms, from 1 to m_max

    Returns
    -------
    summed_term : float
    """
    tm = (
        np.cosh(m * np.pi * (y_D - np.abs(y_i - y_j))) + np.cosh(m * np.pi * (y_D - y_i - y_j))
    ) / np.sinh(m * np.pi * y_D)

    S1 = 2 * np.sum(tm / m * np.cos(m * np.pi * x_i) * np.cos(m * np.pi * x_j))
    tN = tm[-1]
    S2 = -tN / 2 * np.log(
        (1 - np.cos(np.pi * (x_i + x_j))) ** 2 + np.sin(np.pi * (x_i + x_j)) ** 2
    ) - tN / 2 * np.log((1 - np.cos(np.pi * (x_i - x_j))) ** 2 + np.sin(np.pi * (x_i - x_j)) ** 2)
    S3 = -2 * tN * np.sum(1 / m * np.cos(m * np.pi * x_i) * np.cos(m * np.pi * x_j))
    return S1 + S2 + S3
