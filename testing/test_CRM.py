from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from pywaterflood.crm import (
    CRM,
    CrmCompensated,
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


@pytest.fixture()
def reservoir_simulation_data():
    injection = np.genfromtxt(data_dir + "injection.csv", delimiter=",")
    production = np.genfromtxt(data_dir + "production.csv", delimiter=",")
    time = np.genfromtxt(data_dir + "time.csv", delimiter=",")
    return injection, production, time


@pytest.fixture()
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


@pytest.fixture()
def trained_compensated_model(reservoir_simulation_data):
    def _trained_model(*args, **kwargs):
        crm = CrmCompensated(*args, **kwargs)
        crm.injection, crm.production, crm.time = reservoir_simulation_data
        crm.pressure = np.random.default_rng(42).poisson(100, crm.production.shape).astype(float)
        crm.gains = np.genfromtxt(data_dir + "gains.csv", delimiter=",")
        if crm.tau_selection == "per-pair":
            crm.tau = np.genfromtxt(data_dir + "taus_per-pair.csv", delimiter=",")
        else:
            crm.tau = np.genfromtxt(data_dir + "taus.csv", delimiter=",")
        crm.gains_producer = np.genfromtxt(data_dir + "gain_producer.csv", delimiter=",")
        crm.tau_producer = np.genfromtxt(data_dir + "tau_producer.csv", delimiter=",")
        crm.gain_pressure = np.ones([crm.production.shape[1], crm.production.shape[1]], dtype="f8")
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
    _, n_prod = production.shape
    _, n_inj = injection.shape
    gains = random_weights(n_prod, n_inj, seed=42)
    rng = np.random.default_rng(42)
    taus = rng.uniform(1, 40, gains.shape)
    q_1 = np.empty_like(production)
    q_2 = np.empty_like(production)
    for i in range(n_prod):
        q_1[:, i] = q_CRM_perpair(injection, time, gains[i], taus[i])
        q_2[:, i] = q_CRM_perpair(injection, time, 0.5 * gains[i], taus[i])
    assert not np.isnan(q_1).flatten().any(), "No NaNs"
    assert not np.isnan(q_1).flatten().any(), "No NaNs"
    assert np.allclose(q_1, 2.0 * q_2), "Double gains -> double prod"


def test_q_perproducer(reservoir_simulation_data):
    "Secondary production."
    injection, production, time = reservoir_simulation_data
    _, n_prod = production.shape
    _, n_inj = injection.shape
    gains = random_weights(n_prod, n_inj, seed=42)
    rng = np.random.default_rng(42)
    taus = rng.uniform(1, 40, n_prod)
    q_1 = np.empty_like(production)
    q_2 = np.empty_like(production)
    for i in range(n_prod):
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
        with pytest.raises(ValueError, match="constraints"):
            CRM(primary=primary, tau_selection=tau_selection, constraints="negative")
        with pytest.raises(ValueError, match="tau"):
            CRM(primary=primary, tau_selection="per-Bob", constraints=constraints)


@pytest.mark.parametrize("tau_selection", tau_selection)
@pytest.mark.parametrize("primary", primary)
class TestPredict:
    def test_predict(self, reservoir_simulation_data, trained_model, primary, tau_selection):
        injection, _production, time = reservoir_simulation_data
        crm = trained_model(primary=primary, tau_selection=tau_selection)

        prediction1 = crm.predict()
        prediction2 = crm.predict(injection, time)
        prediction3 = crm.predict(connections={"gains": crm.gains})
        assert prediction1 == pytest.approx(prediction2, abs=1.0)
        assert prediction2 == pytest.approx(prediction3, abs=1)

        primary_str = "primary" if primary else "noprimary"

        assert prediction1 == pytest.approx(
            np.genfromtxt(
                f"{data_dir}prediction_{primary_str}_{tau_selection}.csv", delimiter=","
            ),
            abs=5.0,
            rel=1e-2,
        )

    def test_predict_untrained(self, reservoir_simulation_data, primary, tau_selection):
        """Test predicting from gains in as a guess with no prior fitting."""
        crm = CRM(primary=primary, tau_selection=tau_selection)
        injection, production, time = reservoir_simulation_data
        n_inj = injection.shape[1]
        n_prod = production.shape[1]
        if tau_selection == "per-producer":
            tau = np.full(n_prod, 21.1)
        else:
            tau = np.full((n_inj, n_prod), 21.1)
        connections = {
            "gains": np.ones((n_inj, n_prod)),
            "tau": tau,
            "gains_producer": np.ones(n_prod),
            "tau_producer": np.full(n_prod, 20.0),
        }
        prediction = crm.predict(injection, time, connections=connections, production=production)
        assert np.all(prediction >= 0)
        if not primary:
            connections = {
                "gains": np.ones((n_inj, n_prod)),
                "tau": tau,
            }
            prediction = crm.predict(
                injection, time, connections=connections, production=production
            )
            assert np.all(prediction >= 0)

    def test_predict_fails(self, reservoir_simulation_data, trained_model, primary, tau_selection):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary, tau_selection)
        with pytest.raises(
            TypeError, match="Either both or neither of injection or time must be specified"
        ):
            crm.predict(injection)
        with pytest.raises(ValueError, match="number of steps"):
            crm.predict(injection, time[:-1])

    def test_set_connections(
        self, reservoir_simulation_data, trained_model, primary, tau_selection
    ):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary=primary, tau_selection=tau_selection)
        crm2 = CRM(primary=primary, tau_selection=tau_selection)
        crm2.set_connections(
            gains=crm.gains,
            tau=crm.tau,
            gains_producer=crm.gains_producer,
            tau_producer=crm.tau_producer,
        )
        crm2.set_connections()  # no-op, hopefully
        prediction2 = crm2.predict(injection, time, production=production)
        assert crm.predict(injection, time) == pytest.approx(prediction2)
        assert production.shape == prediction2.shape

    def test_residual(self, reservoir_simulation_data, trained_model, primary, tau_selection):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary=primary, tau_selection=tau_selection)
        q_hat = crm.predict(injection, time)
        resid1 = production - q_hat
        resid2 = crm.residual(production, injection, time)
        resid3 = crm.residual()
        assert resid1 == pytest.approx(resid2)
        assert resid1 == pytest.approx(resid3)
        assert resid1.shape == (len(time), production.shape[-1])


@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestFit:
    def test_validate_timeseries(
        self, reservoir_simulation_data, primary, tau_selection, constraints
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        with pytest.raises(ValueError, match="same number"):
            crm.set_rates(production[:-5], injection, time)
        with pytest.raises(ValueError, match="same number"):
            crm.set_rates(production, injection[:-5], time)
        with pytest.raises(ValueError, match="same number"):
            crm.set_rates(production, injection, time[:-5])
        prod_bad = production.copy()
        prod_bad[4, 0] = -1
        with pytest.raises(ValueError, match="negative"):
            crm.set_rates(prod_bad, injection, time)
        prod_bad[4, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            crm.set_rates(prod_bad, injection, time)
        inj_bad = injection.copy()
        inj_bad[2, 2] = -1
        with pytest.raises(ValueError, match="negative"):
            crm.set_rates(production, inj_bad, time)
        inj_bad[2, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            crm.set_rates(production, inj_bad, time)
        time_bad = time.copy()
        time_bad[0] = -1
        with pytest.raises(ValueError, match="negative"):
            crm.set_rates(time=time_bad)
        time_bad[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            crm.set_rates(time=time_bad)

    def test_fit_fails(self, reservoir_simulation_data, primary, tau_selection, constraints):
        crm = CRM(primary, tau_selection, constraints)
        injection, production, time = reservoir_simulation_data

        with pytest.raises(TypeError, match="missing .* positional argument"):
            crm.fit(production)
        with pytest.raises(TypeError, match="missing .* positional argument"):
            crm.fit(production, injection)
        with pytest.raises(ValueError, match="same number of timesteps"):
            crm.fit(production[:-1], injection, time)
        with pytest.raises(ValueError, match="same number of timesteps"):
            crm.fit(production, injection[:-1], time)
        with pytest.raises(ValueError, match="same number of timesteps"):
            crm.fit(production, injection, time[:-1])
        with pytest.raises(ValueError, match="production"):
            crm.fit(production[:-1], injection, None)
        crm.constraints = "sum-to-one injector"
        with pytest.raises(NotImplementedError):
            crm.fit(production, injection, time)
        crm.constraints = ""
        with pytest.raises(ValueError, match="constrain"):
            crm.fit(production, injection, time)

    @pytest.mark.slow()
    @pytest.mark.parametrize("random", [True, False])
    def test_fit_serial(
        self, reservoir_simulation_data, primary, tau_selection, constraints, random
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(
            production,
            injection,
            time,
            num_cores=1,
            options={"maxiter": 3},
            random=random,
        )

    @pytest.mark.parametrize("random", [True, False])
    def test_fit_parallel(
        self, reservoir_simulation_data, primary, tau_selection, constraints, random
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(
            production,
            injection,
            time,
            num_cores=4,
            options={"maxiter": 3},
            random=random,
        )

    @pytest.mark.slow()
    @pytest.mark.parametrize("random", [True, False])
    def test_fit_initial_guess(
        self, reservoir_simulation_data, primary, tau_selection, constraints, random
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.set_rates(production, injection, time)
        crm.set_rates()  # no-op
        x0 = crm._get_initial_guess(tau_selection)
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
        with pytest.raises(TypeError, match="missing .* argument"):
            crm.to_excel()
        crm2 = CRM(primary)
        with pytest.raises(ValueError, match="Model has not been trained"):
            crm2.to_excel(tmpdir + "/test.xlsx")

    def test_to_pickle(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        crm.to_pickle(tmpdir + "/test.pkl")

    def test_to_pickle_fails(self, trained_model, primary):
        crm = trained_model(primary)
        with pytest.raises(TypeError, match="missing .* argument"):
            crm.to_pickle()


@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestCompensated:
    def test_init_compensated(self, primary, tau_selection, constraints):
        crm = CrmCompensated(primary, tau_selection, constraints)
        assert crm.primary == primary

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("num_cores", [1, 4])
    @pytest.mark.parametrize("initial_guess", [True, None])
    def test_fit(
        self,
        reservoir_simulation_data,
        primary,
        tau_selection,
        constraints,
        random,
        num_cores,
        initial_guess,
    ):
        injection, production, time = reservoir_simulation_data
        crm = CrmCompensated(primary, tau_selection, constraints)
        crm.set_rates(production, injection, time)
        pressure = np.ones_like(production)
        if initial_guess:
            initial_guess = crm._get_initial_guess(random=random)
        crm.fit(
            production,
            pressure,
            injection,
            time,
            initial_guess,
            num_cores=num_cores,
            options={"maxiter": 3},
            random=random,
        )
        prediction1 = crm.predict()
        prediction2 = crm.predict(injection, time)
        assert prediction1 == pytest.approx(prediction2)
        prediction3 = crm.predict(injection, time, pressure=pressure)
        assert prediction3 == pytest.approx(prediction1)

    def test_fit_fails(
        self,
        reservoir_simulation_data,
        primary,
        tau_selection,
        constraints,
    ):
        crm = CrmCompensated(primary, tau_selection, constraints)
        injection, production, time = reservoir_simulation_data
        pressure = np.ones_like(production)

        with pytest.raises(ValueError, match="same number of timesteps"):
            crm.fit(production, pressure[:-1], injection, time)
        with pytest.raises(ValueError, match="production and pressure"):
            crm.fit(production, pressure[:-1], injection, None)
        with pytest.raises(ValueError, match="injection and pressure"):
            crm.fit(None, pressure[:-1], injection, None)


@pytest.mark.parametrize("tau_selection", tau_selection)
@pytest.mark.parametrize("primary", primary)
class TestPredictCompensated:
    def test_predict(
        self, reservoir_simulation_data, trained_compensated_model, primary, tau_selection
    ):
        injection, _production, time = reservoir_simulation_data
        crm = trained_compensated_model(primary=primary, tau_selection=tau_selection)

        prediction1 = crm.predict()
        prediction2 = crm.predict(injection, time)
        prediction3 = crm.predict(connections={"gains": crm.gains})
        assert prediction1 == pytest.approx(prediction2, abs=1.0)
        assert prediction2 == pytest.approx(prediction3, abs=1)

        # primary_str = "primary" if primary else "noprimary"
        # assert prediction1 == pytest.approx(
        #     np.genfromtxt(
        #         f"{data_dir}prediction_{primary_str}_{tau_selection}.csv", delimiter=","
        #     ),
        #     abs=5.0,
        #     rel=1e-2,
        # )

    def test_predict_untrained(self, reservoir_simulation_data, primary, tau_selection):
        """Test predicting from gains in as a guess with no prior fitting."""
        crm = CRM(primary=primary, tau_selection=tau_selection)
        injection, production, time = reservoir_simulation_data
        n_inj = injection.shape[1]
        n_prod = production.shape[1]
        if tau_selection == "per-producer":
            tau = np.full(n_prod, 21.1)
        else:
            tau = np.full((n_inj, n_prod), 21.1)
        connections = {
            "gains": np.ones((n_inj, n_prod)),
            "tau": tau,
            "gains_producer": np.ones(n_prod),
            "tau_producer": np.full(n_prod, 20.0),
        }
        prediction = crm.predict(injection, time, connections=connections, production=production)
        assert np.all(prediction >= 0)
        if not primary:
            connections = {
                "gains": np.ones((n_inj, n_prod)),
                "tau": tau,
            }
            prediction = crm.predict(
                injection, time, connections=connections, production=production
            )
            assert np.all(prediction >= 0)

    def test_predict_fails(
        self, reservoir_simulation_data, trained_compensated_model, primary, tau_selection
    ):
        injection, _production, time = reservoir_simulation_data
        crm = trained_compensated_model(primary, tau_selection)
        with pytest.raises(
            TypeError, match="Either both or neither of injection or time must be specified"
        ):
            crm.predict(injection)
        with pytest.raises(ValueError, match="number of steps"):
            crm.predict(injection, time[:-1])

    def test_residual(
        self, reservoir_simulation_data, trained_compensated_model, primary, tau_selection
    ):
        injection, production, time = reservoir_simulation_data
        crm = trained_compensated_model(primary=primary, tau_selection=tau_selection)
        q_hat = crm.predict(injection, time)
        resid1 = production - q_hat
        resid2 = crm.residual(production, injection, time)
        resid3 = crm.residual()
        assert resid1 == pytest.approx(resid2)
        assert resid1 == pytest.approx(resid3)
        assert resid1.shape == (len(time), production.shape[-1])
