import pytest
from itertools import product

from pyCRM import CRM

primary = (True, False)
tau_selection = ('per-pair', 'per-producer')
constraints = ('positive', 'up-to one', 'sum-to-one')
test_args = list(product(primary, tau_selection, constraints))

@pytest.fixture
def reservoir_simulation_data():
    injection = np.genfromtxt('data/injection.csv', delimiter=',')
    production = np.genfromtxt('data/production.csv', delimiter=',')
    time = np.genfromtxt('data/time.csv', delimiter=',')
    return injection, production, time

@pytest.fixture
def trained_model():
    crm = CRM()
    injection, production, time = resrevoir_simulation_data()
    #crm.gains = 
    return crm

@pytest.mark.parametrize("primary,tau_selection,constraints", test_args)
class TestCRM:
    def test_init(self, primary, tau_selection, constraints):
        CRM(primary, tau_selection, constraints)

    
    def test_init_fails(self, primary, tau_selection, constraints):
        with pytest.raises(TypeError):
            CRM(primary='yes', tau_selection=tau_selection, constraints=constraints)
        with pytest.raises(ValueError):
            CRM(primary=primary, tau_selection=tau_selection, constraints = 'negative')
        with pytest.raises(ValueError):
            CRM(primary=primary, tau_selection = 'per-Bob', constraints=constraints)
  
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

    def test_fit_serial(self, reservoir_simulation_data, primary, tau_selection, constraints):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(production, injection, time, num_cores=1)
   
    def test_fit_parallel(self, reservoir_simulation_data, primary, tau_selection, constraints):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        crm.fit(production, injection, time, num_cores=4)

    @pytest.mark.skip
    def test_predict(self, reservoir_simulation_data, primary, tau_selection, constraints):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        
        crm.fit(production, injection, time, num_cores=4)
        crm.predict()
        crm.predict(injection, time)

    def test_predict_fails(self, reservoir_simulation_data, trained_model):
        injection, production, time = reservoir_simulation_data
        crm = CRM(primary, tau_selection, constraints)
        with pytest.raises(AttributeError):
            crm.predict(injection)
            
    def test_to_excel(self, trained_model):
        pass
    
    def test_to_excel_fails(self, trained_model):
        pass
        