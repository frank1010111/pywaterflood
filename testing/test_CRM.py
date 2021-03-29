import pytest
from itertools import product
import numpy as np

import sys
sys.path.append('../pyCRM/')
from pyCRM import CRM

#import CRM

primary = (True, False)
tau_selection = ('per-pair', 'per-producer')
constraints = ('positive', 'up-to one', 'sum-to-one', 'sum-to-one injector')
random = (True, False)
test_args = list(product(primary, tau_selection, constraints))
#fit_args = list(product(primary, tau_selection, constraints, random))
data_dir = 'testing/data/'

@pytest.fixture
def reservoir_simulation_data():
    injection = np.genfromtxt(data_dir + 'injection.csv', delimiter=',')
    production = np.genfromtxt(data_dir + 'production.csv', delimiter=',')
    time = np.genfromtxt(data_dir + 'time.csv', delimiter=',')
    return injection, production, time

@pytest.fixture
def trained_model(reservoir_simulation_data):
    def _trained_model(*args, **kwargs):
        crm = CRM(*args, **kwargs)
        injection, production, time = reservoir_simulation_data
        crm.set_rates(production, injection, time)
        gains = np.genfromtxt(data_dir + 'gains.csv', delimiter=',')
        if crm.tau_selection == 'per-pair':
            tau = np.genfromtxt(data_dir + 'taus_per-pair.csv', delimiter=',')
        else:
            tau = np.genfromtxt(data_dir + 'taus.csv', delimiter=',')
        gains_producer = np.genfromtxt(data_dir + 'gain_producer.csv', delimiter=',')
        tau_producer = np.genfromtxt(data_dir + 'tau_producer.csv', delimiter=',')
        crm.set_connections(gains, tau, gains_producer, tau_producer)
        return crm
    return _trained_model

@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestInstantiate:
    def test_init(self, primary, tau_selection, constraints):
        CRM(primary, tau_selection, constraints)


    def test_init_fails(self, primary, tau_selection, constraints):
        with pytest.raises(TypeError):
            CRM(primary='yes', tau_selection=tau_selection, constraints=constraints)
        with pytest.raises(ValueError):
            CRM(primary=primary, tau_selection=tau_selection, constraints = 'negative')
        with pytest.raises(ValueError):
            CRM(primary=primary, tau_selection = 'per-Bob', constraints=constraints)



@pytest.mark.parametrize("tau_selection", tau_selection)
@pytest.mark.parametrize("primary", primary)
class TestPredict:
    def test_set_rates(self, reservoir_simulation_data, primary, tau_selection):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection)
        crm.set_rates(production)
        crm.set_rates(production, injection)
        crm.set_rates(production, injection, time)

    def test_predict(self, reservoir_simulation_data, trained_model, primary, tau_selection):
        injection, production, time = reservoir_simulation_data
        crm = trained_model(primary=primary, tau_selection=tau_selection)

        prediction1 = crm.predict()
        prediction2 = crm.predict(injection, time)
        prediction1 == pytest.approx(prediction2)

        if primary:
            prediction1 == pytest.approx(
                np.genfromtxt(data_dir + 'prediction.csv', delimiter=',')
            )
        else:
            prediction1 == pytest.approx(
                np.genfromtxt(data_dir + 'prediction_noprimary.csv', delimiter=',')
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
            crm.fit(production, injection, time[:-1],)

    @pytest.mark.slow
    @pytest.mark.parametrize("random", random)
    def test_fit(self, reservoir_simulation_data, primary, tau_selection, constraints, random):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(production, injection, time, random=random, options={'maxiter':100})


@pytest.mark.parametrize("primary", primary)
class TestExport:
    def test_to_excel(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        crm.to_excel(tmpdir + '/test.xlsx')

    def test_to_excel_fails(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        with pytest.raises(TypeError):
            crm.to_excel()
        crm2 = CRM(primary)
        with pytest.raises(ValueError):
            crm2.to_excel(tmpdir + '/test.xlsx')

    def test_to_pickle(self, trained_model, primary, tmpdir):
        crm = trained_model(primary)
        crm.to_pickle(tmpdir + '/test.pkl')

    def test_to_pickle_fails(self, trained_model, primary):
        crm = trained_model(primary)
        with pytest.raises(TypeError):
            crm.to_pickle()
