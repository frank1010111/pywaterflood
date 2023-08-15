from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pywaterflood.multiwellproductivity import (
    calc_A_ij,
    calc_gains_homogeneous,
    calc_influence_matrix,
    translate_locations,
)

raw_locs = pd.DataFrame()


@pytest.fixture()
def locations():
    "Well locations for an inverted 5-spot."
    xy = [(1, 1), (20, 1), (1, 20), (20, 20), (10.5, 10.5)]
    x = np.array([x for x, _ in xy]) + np.random.normal(0, 0.5, len(xy))
    y = np.array([y for _, y in xy]) + np.random.normal(0, 0.5, len(xy))
    locations = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "Type": ["Producer"] * 4 + ["Injector"],
        }
    )
    return translate_locations(locations, "x", "y", "Type")


def test_translate(locations):
    locations = translate_locations(locations, "X", "Y", "Type")
    assert locations["X"].min() <= 1e-10
    assert locations["Y"].min() <= 1e-10


def test_Aij():
    """From the worked example by Valkó et al, 2000.

    Development and Application of the Multiwell Productivity Index (MPI)"""
    m = 1 + np.arange(300, dtype="uint64")
    results = calc_A_ij(0.233351184, 0.36666667, 0.23333333, 0.36666667, 0.5, m)
    assert pytest.approx(10.6867, abs=0.01) == results
    for x_i in [0.0, 0.2, 0.3, 0.5]:
        old_A_ij = calc_A_ij_old(x_i, 0.36666667, 0.23333333, 0.36666667, 0.5, m)
        rust_A_ij = calc_A_ij(x_i, 0.36666667, 0.23333333, 0.36666667, 0.5, m)
        assert pytest.approx(old_A_ij, 1e-3) == rust_A_ij


def test_Aij_symmetry():
    idx = pd.IndexSlice
    m_max = 100
    locations = pd.DataFrame({"X": [0.2, 0.8], "Y": [0.3, 0.7]})
    y_D = 1.1
    influence_matrix = pd.DataFrame(
        index=pd.MultiIndex.from_product([locations.index, locations.index]), columns=["A"]
    )
    m = np.arange(1, m_max + 1, dtype="uint64")  # elements of sum
    for i, j in influence_matrix.index:
        x_i, y_i = locations.loc[i, ["X", "Y"]]
        x_j, y_j = locations.loc[j, ["X", "Y"]] + 1e-6
        influence_matrix.loc[idx[i, j], "A"] = calc_A_ij(x_i, y_i, x_j, y_j, y_D, m)
    influence_matrix = influence_matrix["A"].unstack().astype("float64")
    assert pytest.approx(influence_matrix.iloc[0, 1], 1e-4) == influence_matrix.iloc[1, 0]


def test_calc_gains(locations):
    """See that calc_gains works."""
    x_e = 34
    y_e = 22
    gains = calc_gains_homogeneous(locations, x_e, y_e)
    assert np.all(gains <= 0), "Gains must be negative"
    assert np.all(gains >= -1), "Gains can't be stronger than one"


def test_influence_matrix(locations):
    locations["X"] /= 20
    locations["Y"] /= 20
    matrix_conn = calc_influence_matrix(locations, y_D=0.7, matrix_type="conn", m_max=100)
    assert not np.isnan(matrix_conn.values).flatten().any()

    matrix_prod = calc_influence_matrix(locations, y_D=0.7, matrix_type="prod", m_max=100)
    assert not np.isnan(matrix_prod.values).flatten().any()
    with pytest.raises(ValueError, match="matrix_type must be"):
        calc_influence_matrix(locations, y_D=0.7, matrix_type="wrong")


# for regression testing
def calc_A_ij_old(x_i: float, y_i: float, x_j: float, y_j: float, y_D: float, m: NDArray) -> float:
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
    y_j, y_i = sorted([y_i, y_j])
    x_j, x_i = sorted([x_i, x_j])
    if not ((x_i - x_j) > (y_i - y_j)):
        y_D = 1 / y_D
    first_term = 2 * np.pi * y_D * (1 / 3.0 - y_i / y_D + (y_i**2 + y_j**2) / (2 * y_D**2))

    tm = (
        np.cosh(m * np.pi * (y_D - np.abs(y_i - y_j))) + np.cosh(m * np.pi * (y_D - y_i - y_j))
    ) / np.sinh(m * np.pi * y_D)

    S1 = 2 * np.sum(tm / m * np.cos(m * np.pi * x_i) * np.cos(m * np.pi * x_j))
    tN = tm[-1]
    S2 = -tN / 2 * np.log(
        (1 - np.cos(np.pi * (x_i + x_j))) ** 2 + np.sin(np.pi * (x_i + x_j)) ** 2
    ) - tN / 2 * np.log((1 - np.cos(np.pi * (x_i - x_j))) ** 2 + np.sin(np.pi * (x_i - x_j)) ** 2)
    S3 = -2 * tN * np.sum(1 / m * np.cos(m * np.pi * x_i) * np.cos(m * np.pi * x_j))
    summed_term = S1 + S2 + S3

    return first_term + summed_term
