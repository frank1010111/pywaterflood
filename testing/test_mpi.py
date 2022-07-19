from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywaterflood.multiwellproductivity import (
    calc_A_ij,
    calc_gains_homogeneous,
    calc_influence_matrix,
    translate_locations,
)

raw_locs = pd.DataFrame()


@pytest.fixture
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
    locations = translate_locations(locations, "x", "y", "Type")
    return locations


def test_translate(locations):
    locations = translate_locations(locations, "X", "Y", "Type")
    assert locations["X"].min() <= 1e-10
    assert locations["Y"].min() <= 1e-10


def test_Aij():
    "From the worked example by Kaviani and Valkó"
    m = 1 + np.arange(300)
    results = calc_A_ij(0.233351184, 0.36666667, 0.23333333, 0.36666667, 0.5, m)
    assert pytest.approx(10.6867, abs=0.01) == results


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
