from __future__ import annotations

import math
from copy import copy
from typing import Literal

import numpy as np
import pytest
from pywaterflood.aquifer import (
    aquifer_production,
    effective_reservoir_radius,
    get_bessel_roots,
    klins_pressure_dimensionless,
    klins_water_dimensionless_finite,
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
def test_water_aquifer_prod(method: Literal["marsal-walsh"] | Literal["klins"]):
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
def test_water_dimensionless(method: Literal["marsal-walsh"] | Literal["klins"]):
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


def test_get_bessel_roots():
    """Compare the numerically calculated Bessel roots with the regression.

    Source
    ------
    Klins, M. A., Bouchard, A. J., and C. L. Cable.
    "A Polynomial Approach to the van Everdingen-Hurst Dimensionless Variables for
    Water Encroachment." SPE Res Eng 3 (1988): 320-326.
    doi: https://doi.org/10.2118/15433-PA
    """
    b_alpha1 = [-0.00222107, -0.627638, 6.277915, -2.734405, 1.2708, -1.100417]
    b_alpha2 = [-0.00796608, -1.85408, 18.71169, -2.758326, 4.829162, -1.009021]
    b_beta1 = [-0.00870415, -1.08984, 12.4458, -2.8446, 3.4234, -0.949162]
    b_beta2 = [-0.0191642, -2.47644, 25.3343, -2.73054, 6.13184, -0.939529]
    for r_ed in (2, 5, 10, 15, 20):
        alpha1 = (
            b_alpha1[0]
            + b_alpha1[1] / math.sinh(r_ed)
            + b_alpha1[2] * (r_ed) ** b_alpha1[3]
            + b_alpha1[4] * r_ed ** b_alpha1[5]
        )
        alpha2 = (
            b_alpha2[0]
            + b_alpha2[1] / math.sinh(r_ed)
            + b_alpha2[2] * (r_ed) ** b_alpha2[3]
            + b_alpha2[4] * r_ed ** b_alpha2[5]
        )
        alpha_fit = get_bessel_roots(r_ed, 2, "alpha")
        assert pytest.approx(np.array([alpha1, alpha2]), rel=1e-2) == alpha_fit

        beta1 = (
            b_beta1[0]
            + b_beta1[1] / math.sinh(r_ed)
            + b_beta1[2] * r_ed ** b_beta1[3]
            + b_beta1[4] * r_ed ** b_beta1[5]
        )
        beta2 = (
            b_beta2[0]
            + b_beta2[1] / math.sinh(r_ed)
            + b_beta2[2] * r_ed ** b_beta2[3]
            + b_beta2[4] * r_ed ** b_beta2[5]
        )
        beta_fit = get_bessel_roots(r_ed, 2, "beta")
        assert pytest.approx(np.array([beta1, beta2]), rel=1e-2) == beta_fit


def test_bessel_fails():
    with pytest.raises(ValueError, match="root choice"):
        get_bessel_roots(1.5, 2, "charlie")


def test_klins_pressure():
    """Test Klins Appendix G."""
    t_d = 20.0
    r_ed = 10.0
    p_d = klins_pressure_dimensionless(t_d, r_ed)
    assert pytest.approx(1.9690, rel=1e-3) == p_d


def test_test_water_dimensionless_finite_klins():
    """Follow Appendix H from Klins."""
    t_d = 20.0
    r_ed = 10.0
    q_d = klins_water_dimensionless_finite(t_d, r_ed, 2)
    assert pytest.approx(12.2640, rel=1e-2) == q_d
