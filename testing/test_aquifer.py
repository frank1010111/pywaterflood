from __future__ import annotations

import math
from copy import copy

import numpy as np
import pytest
from pywaterflood.aquifer import (
    aquifer_production,
    effective_reservoir_radius,
    water_dimensionless,
    water_dimensionless_infinite,
)

valid_reservoir_inputs = {
    "reservoir_pore_volume": 10.0,
    "porosity": 0.5,
    "thickness": 1.0,
    "flow_solid_angle": 69,
}
invalid_reservoir_inputs = [
    ("reservoir_pore_volume", -1.0),
    ("porosity", -0.4),
    ("porosity", 1.2),
    ("thickness", -2),
    ("flow_solid_angle", -10),
    ("flow_solid_angle", 399),
]


def test_effective_radius():
    """Regression test radius"""
    ex = pytest.approx(math.sqrt(5.6146 / math.pi))
    assert ex == effective_reservoir_radius(1, 1, 1, 360), "Effective radius calculation is wonky"


@pytest.mark.parametrize(("var", "invalid_arg"), invalid_reservoir_inputs)
def test_effective_radius_checks(var, invalid_arg):
    """Test that effective radius remains sane."""
    test_inputs = copy(valid_reservoir_inputs)
    test_inputs[var] = invalid_arg
    match = "domain" if invalid_arg < 0 else var
    with pytest.raises(ValueError, match=match):
        effective_reservoir_radius(**test_inputs)


@pytest.mark.parametrize("method", ["marsal-walsh", "klins"])
def test_water_aquifer_prod(method):
    n_times = 20
    t_d = np.arange(n_times, dtype=np.float64)
    delta_pressure = np.random.default_rng(3245).normal(0, 0.25, n_times)
    for r_ed in [1.1, 3, 40]:
        aquifer_production(
            delta_pressure,
            t_d,
            r_ed,
            method=method,
            aquifer_constant=1.0,
        )


@pytest.mark.parametrize("method", ["marsal-walsh", "klins"])
def test_water_dimensionless(method):
    t_d = 5
    r_ed_infinite = 1e9
    wd_finite = water_dimensionless(t_d, r_ed_infinite, method)
    wd_infinite = water_dimensionless_infinite(t_d, method)
    assert (
        pytest.approx(wd_finite) == wd_infinite
    ), "for nearly infinite effective radius, assume no difference"

    t_d2 = np.linspace(1, 200)
    r_ed2 = 2.0
    wd_finite = water_dimensionless(t_d2, r_ed2, method)
    wd_infinite = water_dimensionless_infinite(t_d2, method)
    assert np.sum(np.isnan(wd_finite)) == 0, "No NaNs for finite Wd"
    assert np.sum(np.isnan(wd_infinite)) == 0, "No NaNs for infinite Wd"
    assert (
        sum(wd_finite > wd_infinite) == 0
    ), "finite reservoir should never be bigger than infinite"
    assert sum(wd_infinite > wd_finite) > (
        len(wd_finite) / 2
    ), "finite reservoir should be generally smaller than infinite"

    with pytest.raises(ValueError, match="r_ed"):
        water_dimensionless(t_d2, r_ed=0.9, method=method)


def test_water_dimensionless_invalid():
    with pytest.raises(ValueError, match="method must be"):
        water_dimensionless(3.0, 3.0, method="korval")
