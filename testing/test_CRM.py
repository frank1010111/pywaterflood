from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from pywaterflood.crm import (
    CRM,
    CrmCompensated,
    _pressure_diff,
    q_bhp,
    q_CRM_perpair,
    q_CRM_perproducer,
    q_primary,
    random_weights,
)

primary = (True, False)
tau_selection = ("per-pair", "per-producer")
constraints = ("positive", "up-to one", "sum-to-one")
test_args = list(product(primary, tau_selection, constraints))
data_dir = "testing/data/"


@pytest.fixture
def reservoir_simulation_data():
    injection = np.genfromtxt(data_dir + "injection.csv", delimiter=",")
    production = np.genfromtxt(data_dir + "production.csv", delimiter=",")
    time = np.genfromtxt(data_dir + "time.csv", delimiter=",")
    return injection, production, time


@pytest.fixture
def trained_model(reservoir_simulation_data):
    def _trained_model(*args, **kwargs):
        crm = CRM(*args, **kwargs)
        crm.injection, crm.production, crm.time = reservoir_simulation_data
        crm.gains = np.genfromtxt(data_dir + "gains.csv", delimiter=",")
        if crm.tau_selection == "per-pair":
            crm.tau = np.genfromtxt(data_dir + "taus_per-pair.csv", delimiter=",")
        else:
            crm.tau = np.genfromtxt(data_dir + "taus.csv", delimiter=",")
        crm.gains_producer = np.genfromtxt(data_dir + "gain_producer.csv", delimiter=",")
        crm.tau_producer = np.genfromtxt(data_dir + "tau_producer.csv", delimiter=",")
        return crm

    return _trained_model


def test_q_primary(reservoir_simulation_data):
    "primary production."
    _, production, time = reservoir_simulation_data
    _, nprod = production.shape
    rng = np.random.default_rng(14_210)
    gain_producer = rng.uniform(0, 1, nprod)
    tau_producer = rng.uniform(1, 100, nprod)
    q = np.empty_like(production)
    for i in range(nprod):
        q[:, i] = q_primary(production[i], time, gain_producer[i], tau_producer[i])
    assert not np.isnan(q).flatten().any(), "No NaNs"
    pxg = production[0] * gain_producer
    summed_q = q.sum(0)
    assert np.allclose(pxg.argsort(), summed_q.argsort()), "Higher gains -> higher predictions"


def test_q_perpair(reservoir_simulation_data):
    "Secondary production."
    injection, production, time = reservoir_simulation_data
    _, nprod = production.shape
    _, ninj = injection.shape
    gains = random_weights(nprod, ninj, seed=42)
    rng = np.random.default_rng(42)
    taus = rng.uniform(1, 40, gains.shape)
    q_1 = np.empty_like(production)
    q_2 = np.empty_like(production)
    for i in range(nprod):
        q_1[:, i] = q_CRM_perpair(injection, time, gains[i], taus[i])
        q_2[:, i] = q_CRM_perpair(injection, time, 0.5 * gains[i], taus[i])
    assert not np.isnan(q_1).flatten().any(), "No NaNs"
    assert not np.isnan(q_1).flatten().any(), "No NaNs"
    assert np.allclose(q_1, 2.0 * q_2), "Double gains -> double prod"


def test_q_perproducer(reservoir_simulation_data):
    "Secondary production."
    injection, production, time = reservoir_simulation_data
    _, nprod = production.shape
    _, ninj = injection.shape
    gains = random_weights(nprod, ninj, seed=42)
    rng = np.random.default_rng(42)
    taus = rng.uniform(1, 40, nprod)
    q_1 = np.empty_like(production)
    q_2 = np.empty_like(production)
    for i in range(nprod):
        q_1[:, i] = q_CRM_perproducer(injection, time, gains[i], taus[i])
        q_2[:, i] = q_CRM_perproducer(injection, time, 0.5 * gains[i], taus[i])
    assert not np.isnan(q_1).flatten().any(), "No NaNs"
    assert not np.isnan(q_1).flatten().any(), "No NaNs"
    assert np.allclose(q_1, 2.0 * q_2), "Double gains -> double prod"


def test_q_bhp_nonan(reservoir_simulation_data):
    "q_BHP functions"
    production = reservoir_simulation_data[1]
    _, nprod = production.shape
    pressure = production.copy()
    rng = np.random.default_rng(42)
    pressure_diff = _pressure_diff(pressure[:, 0], pressure)
    assert not np.isnan(pressure_diff).flatten().any(), "No NaNs among pressure_diff"
    producer_gains = rng.normal(0, 1, nprod)
    q = q_bhp(pressure[:, 0], pressure, producer_gains)
    assert not np.isnan(q).any(), "No NaNs among results"


def test_q_bhp():
    """Test bottomhole pressure influencing production."""
    n_time = 5
    n_prod = 2
    pressure = np.ones((n_time, n_prod))
    producer_gains = np.random.rand(n_prod)
    q = q_bhp(pressure[:, 0], pressure, producer_gains)
    assert np.allclose(q, 0), "no pressure change -> no prod"
    pressure[-1] = 0
    producer_gains = np.ones(n_prod)
    q = q_bhp(pressure[:, 0], pressure, producer_gains)
    assert np.allclose(q[-1], 2), "pressure drop -> increase production"
    pressure[-1] = 2
    producer_gains = np.ones(n_prod)
    q = q_bhp(pressure[:, 0], pressure, producer_gains)
    assert np.allclose(q[-1], -2), "pressure increase -> drop production"


@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestInstantiate:
    def test_init(self, primary, tau_selection, constraints):
        CRM(primary, tau_selection, constraints)

    def test_init_fails(self, primary, tau_selection, constraints):
        with pytest.raises(TypeError):
            CRM(primary="yes", tau_selection=tau_selection, constraints=constraints)
        with pytest.raises(ValueError):
            CRM(primary=primary, tau_selection=tau_selection, constraints="negative")
        with pytest.raises(ValueError):
            CRM(primary=primary, tau_selection="per-Bob", constraints=constraints)


@pytest.mark.parametrize("tau_selection", tau_selection)
@pytest.mark.parametrize("primary", primary)
class TestPredict:
    def test_predict(self, reservoir_simulation_data, trained_model, primary, tau_selection):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary=primary, tau_selection=tau_selection)

        prediction1 = crm.predict()
        prediction2 = crm.predict(injection, time)
        assert prediction1 == pytest.approx(prediction2, abs=1.0)

        if primary:
            primary_str = "primary"
        else:
            primary_str = "noprimary"

        assert prediction1 == pytest.approx(
            np.genfromtxt(
                f"{data_dir}prediction_{primary_str}_{tau_selection}.csv", delimiter=","
            ),
            abs=5.0,
            rel=1e-2,
        )

    def test_predict_fails(self, reservoir_simulation_data, trained_model, primary, tau_selection):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary, tau_selection)
        with pytest.raises(TypeError):
            crm.predict(injection)
        with pytest.raises(ValueError):
            crm.predict(injection, time[:-1])


@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestFit:
    def test_validate_timeseries(
        self, reservoir_simulation_data, primary, tau_selection, constraints
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        with pytest.raises(ValueError):
            crm.set_rates(production[:-5], injection, time)
        with pytest.raises(ValueError):
            crm.set_rates(production, injection[:-5], time)
        with pytest.raises(ValueError):
            crm.set_rates(production, injection, time[:-5])
        prod_bad = production.copy()
        prod_bad[4, 0] = -1
        with pytest.raises(ValueError):
            crm.set_rates(prod_bad, injection, time)
        prod_bad[4, 0] = np.nan
        with pytest.raises(ValueError):
            crm.set_rates(prod_bad, injection, time)
        inj_bad = injection.copy()
        inj_bad[2, 2] = -1
        with pytest.raises(ValueError):
            crm.set_rates(production, inj_bad, time)
        inj_bad[2, 2] = np.nan
        with pytest.raises(ValueError):
            crm.set_rates(production, inj_bad, time)
        time_bad = time.copy()
        time_bad[0] = -1
        with pytest.raises(ValueError):
            crm.set_rates(time=time_bad)
        time_bad[0] = np.nan
        with pytest.raises(ValueError):
            crm.set_rates(time=time_bad)

    def test_fit_fails(self, reservoir_simulation_data, primary, tau_selection, constraints):
        crm = CRM(primary, tau_selection, constraints)
        injection, production, time = reservoir_simulation_data

        with pytest.raises(TypeError):
            crm.fit(production)
        with pytest.raises(TypeError):
            crm.fit(production, injection)
        with pytest.raises(ValueError):
            crm.fit(production[:-1], injection, time)
        with pytest.raises(ValueError):
            crm.fit(production, injection[:-1], time)
        with pytest.raises(ValueError):
            crm.fit(production, injection, time[:-1])

    @pytest.mark.slow
    def test_fit_serial(self, reservoir_simulation_data, primary, tau_selection, constraints):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(
            production,
            injection,
            time,
            num_cores=1,
            options={"maxiter": 3},
        )

    def test_fit_parallel(self, reservoir_simulation_data, primary, tau_selection, constraints):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(
            production,
            injection,
            time,
            num_cores=4,
            options={"maxiter": 3},
        )

    @pytest.mark.slow
    def test_fit_initial_guess(
        self, reservoir_simulation_data, primary, tau_selection, constraints
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.set_rates(production, injection, time)
        x0 = crm._get_initial_guess(tau_selection)
        for random in (True, False):
            crm.fit(
                production,
                injection,
                time,
                random=random,
                initial_guess=x0,
                options={"maxiter": 3},
            )


@pytest.mark.parametrize("primary", primary)
class TestExport:
    def test_to_excel(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        crm.to_excel(tmpdir + "/test.xlsx")

    def test_to_excel_fails(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        with pytest.raises(TypeError):
            crm.to_excel()
        crm2 = CRM(primary)
        with pytest.raises(ValueError):
            crm2.to_excel(tmpdir + "/test.xlsx")

    def test_to_pickle(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        crm.to_pickle(tmpdir + "/test.pkl")

    def test_to_pickle_fails(self, trained_model, primary):
        crm = trained_model(primary)
        with pytest.raises(TypeError):
            crm.to_pickle()


@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestBhp:
    def test_init_bhp(self, primary, tau_selection, constraints):
        crm = CrmCompensated(primary, tau_selection, constraints)
        assert crm.primary == primary
