import pytest
from itertools import product

from CRM import *

primary = (True, False)
tau_selection = ('per-pair', 'per-producer')
constraints = ('positive', 'up-to one', 'sum-to-one')

class TestCRM:
    def test_init(self):
        CRM()
        for x, y, z in product(primary, tau_selection, constraints):
            CRM(x, y, z)

    def test_init_fails(self):
        with pytest.raises(TypeError):
            CRM(primary = 'yes')
        with pytest.raises(ValueError):
            CRM(constraints = 'negative')
        with pytest.raises(ValueError):
            CRM(tau_selection = 'per-Bob')

    def test_fit(self):
        injection = np.genfromtxt('../data/injection.csv', delimiter=',')
        production = np.genfromtxt('../data/production.csv', delimiter=',')
        time = np.genfromtxt('../data/time.csv', delimiter=',')
        
        crm_models = [CRM(x, y, z) for x, y, z in
                      product(primary, tau_selection, constraints) ]
        for crm in crm_models:
            with pytest.raises(TypeError):
                crm.fit(production)
            with pytest.raises(TypeError):
                crm.fit(production, injection)
            with pytest.raises(ValueError):
                crm.fit(production, injection, time, num_cores=-1)
            
        for crm in crm_models:
            crm.fit(production, injection, time, num_cores=1)
            crm.fit(production, injection, time, num_cores=4)

    def test_predict(self):
        injection = np.genfromtxt('../data/injection.csv', delimiter=',')
        production = np.genfromtxt('../data/production.csv', delimiter=',')
        time = np.genfromtxt('../data/time.csv', delimiter=',')
        crm_models = [CRM(x, y, z) for x, y, z in
                      product(primary, tau_selection, constraints) ]
        
        for crm in crm_models:
            crm.fit(production, injection, time, num_cores=4)
            crm.predict()
            crm.predict(injection, time)
            with pytest.raises(TypeError):
                crm.predict(injection)
        