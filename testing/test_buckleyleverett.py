from __future__ import annotations

from copy import copy

import numpy as np
import pytest
from pywaterflood.buckleyleverett import Reservoir, breakthrough_sw, water_front_velocity

valid_reservoir_inputs = {
    "phi": 0.5,
    "flow_cross_section": 1.0,
    "viscosity_oil": 4.2e-3,
    "viscosity_water": 1e-3,
    "sat_oil_r": 0.2,
    "sat_water_c": 0.1,
    "sat_gas_c": 0.1,
    "n_oil": 2,
    "n_water": 2,
}
invalid_reservoir_inputs = [
    ("phi", -1.0),
    ("phi", 1.2),
    ("flow_cross_section", -2),
    ("viscosity_oil", -3),
    ("viscosity_water", -4),
    ("sat_oil_r", -5),
    ("sat_oil_r", 1.2),
    ("sat_water_c", -6),
    ("sat_water_c", 3),
    ("sat_gas_c", -7),
    ("sat_gas_c", 4),
    ("n_oil", -1),
    ("n_oil", 0.5),
    ("n_oil", 8),
    ("n_water", 0),
    ("n_water", 9),
]


def test_reservoir_valid():
    valid_reservoir = Reservoir(**valid_reservoir_inputs)
    for key in valid_reservoir_inputs:
        assert valid_reservoir_inputs[key] == getattr(valid_reservoir, key)


@pytest.mark.parametrize(("var", "invalid_arg"), invalid_reservoir_inputs)
def test_reservoir_invalid(var, invalid_arg):
    test_inputs = copy(valid_reservoir_inputs)
    test_inputs[var] = invalid_arg
    with pytest.raises(ValueError, match=var):
        Reservoir(**test_inputs)


def test_reservoir_bad_residual_saturations():
    test_inputs = copy(valid_reservoir_inputs)
    test_inputs["sat_gas_c"] = 0.9
    test_inputs["sat_water_c"] = 0.8
    with pytest.raises(ValueError, match="saturations"):
        Reservoir(**test_inputs)


def test_water_front_velocity():
    reservoir = Reservoir(**valid_reservoir_inputs)
    n_runs = 100
    sat_water_dist = np.random.uniform(0, 1, n_runs)
    flow_rate_dist = np.random.uniform(0, 1e3, n_runs)
    front_v = np.array(
        [
            water_front_velocity(reservoir, sat_water_dist[i], flow_rate_dist[i])
            for i in range(n_runs)
        ]
    )
    assert min(front_v) >= 0, "Front velocity is never negative"


def test_breakthrough_sw():
    n_runs = 100
    rng = np.random.default_rng(42)
    valid_reservoir_dist = {
        "phi": rng.uniform(0, 1, n_runs),
        "flow_cross_section": rng.lognormal(3, 1, n_runs),
        "viscosity_oil": rng.uniform(1e-3, 0.1, n_runs),
        "viscosity_water": rng.uniform(9e-4, 1.1e-3, n_runs),
        "sat_oil_r": rng.beta(2, 7, n_runs),
        "sat_water_c": rng.beta(2, 7, n_runs),
        "sat_gas_c": rng.beta(1, 10, n_runs),
        "n_oil": rng.uniform(1, 6, n_runs),
        "n_water": rng.uniform(1, 6, n_runs),
    }
    for i_run in range(n_runs):
        test_inputs = {key: valid_reservoir_dist[key][i_run] for key in valid_reservoir_dist}
        if test_inputs["sat_oil_r"] + test_inputs["sat_water_c"] + test_inputs["sat_gas_c"] > 0:
            test_inputs["sat_oil_r"] /= 3
            test_inputs["sat_gas_c"] /= 3
            test_inputs["sat_water_c"] /= 3

        reservoir = Reservoir(**test_inputs)
        breakthrough_sw_calc = breakthrough_sw(reservoir)
        assert 0 < breakthrough_sw_calc < 1, "breakthrough saturation must be physical"
        assert (
            test_inputs["sat_water_c"] <= breakthrough_sw_calc
        ), "breakthrough saturation should be above connate water saturation"
        assert (
            1 - test_inputs["sat_oil_r"]
        ) > breakthrough_sw_calc, (
            "breakthrough saturation should be below 1 - residual oil saturation"
        )
