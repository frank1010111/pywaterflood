import numpy as np
import pytest

from itertools import product
from pywaterflood import CRM
from pywaterflood.crm import CrmCompensated, q_BHP

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
        crm.gains_producer = np.genfromtxt(
            data_dir + "gain_producer.csv", delimiter=","
        )
        crm.tau_producer = np.genfromtxt(data_dir + "tau_producer.csv", delimiter=",")
        return crm

    return _trained_model


def test_BHP(reservoir_simulation_data):
    "q_BHP functions"
    production = reservoir_simulation_data[1]
    _, nprod = production.shape
    pressure_test = production.copy()
    rng = np.random.default_rng(42)
    v_matrix = rng.normal(0, 1, (nprod, nprod))
    q = q_BHP(pressure_test, v_matrix)
    assert not np.isnan(q).any()


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
    def test_predict(
        self, reservoir_simulation_data, trained_model, primary, tau_selection
    ):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary=primary, tau_selection=tau_selection)

        prediction1 = crm.predict()
        prediction2 = crm.predict(injection, time)
        assert prediction1 == pytest.approx(prediction2, abs=1.0)

        if primary:
            assert prediction1 == pytest.approx(
                np.genfromtxt(data_dir + "prediction.csv", delimiter=",")
            )
        else:
            assert prediction1 == pytest.approx(
                np.genfromtxt(data_dir + "prediction_noprimary.csv", delimiter=","),
                abs=1.0,
                rel=1e-3,
            )

    def test_predict_fails(
        self, reservoir_simulation_data, trained_model, primary, tau_selection
    ):
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

    def test_fit_fails(
        self, reservoir_simulation_data, primary, tau_selection, constraints
    ):
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

    @pytest.mark.skip
    def test_fit_serial(
        self, reservoir_simulation_data, primary, tau_selection, constraints
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(production, injection, time, num_cores=1)

    def test_fit_parallel(
        self, reservoir_simulation_data, primary, tau_selection, constraints
    ):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(production, injection, time, num_cores=4)

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
                options={"maxiter": 10},
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
