import numpy as np
from numba import jit, njit, prange
import pandas as pd
import pickle
from scipy import optimize
#from joblib import Parallel, delayed

@njit
def q_primary(production, time, gain_producer, tau_producer):
    q_hat = np.zeros_like(production)
    
    # Compute primary production component
    for i in range(production.shape[1]):
        for k in range(len(time)):
            time_decay = np.exp(- time[k] / tau_producer[i])
            q_hat[k, i] += gain_producer[i] * production[0, i] * time_decay
    return q_hat

#@njit
def q_CRM_perproducer(n_producers, injection, time, gains, tau):
    tau2 =  tau[:, np.newaxis] * np.ones((n_producers, injection.shape[1]))
    return q_CRM_perpair(n_producers, injection, time, gains, tau2)

@njit
def q_CRM_perpair(n_producers, injection, time, gains, tau):
    #time_diff_matrix = np.zeros( (len(time), len(time) + 1))
    #for k in range(1, len(time)):
    #    for l in range(1, k+1):
    #        time_diff_matrix[k, l] = time[l] - time[k]
    q_hat = np.zeros((len(time), n_producers))
    # Compute convolved injection rates
    conv_injected = np.zeros( (len(time), injection.shape[1], n_producers) )
    #breakpoint()
    for i in range(n_producers):
        for j in range(injection.shape[1]):
            conv_injected[0, j, i] += (1 - np.exp((time[0] - time[1]) / tau[i, j])) * injection[0, j]
            
            for k in range(1, len(time)):
                for l in range(1,k+1):
                    time_decay = ((1 - np.exp((time[l - 1] - time[l]) / tau[i, j])) # time between l and l-1
                                  * np.exp((time[l] - time[k]) / tau[i, j]) # time between l and k
                                 )
                    conv_injected[k, j, i] += time_decay * injection[l, j]

    # Calculate waterflood rates
    for i in range(n_producers):
        for k in range(len(time)):
            for j in range(injection.shape[1]):
                q_hat[k, i] += gains[i, j] * conv_injected[k, j, i]
    return q_hat

class CRM():
    """A Capacitance Resistance Model history matcher

    CRM uses a physics-inspired mass balance approach to explain production for waterfloods.
    It treants each injector-producer well pair as a system with mass input, output, and pressure
    related to the mass balance. Several versions exist. For an exhaustive review, check
    "A State-of-the-Art Literature Review on Capacitance Resistance Models for Reservoir
    Characterization and Performance Forecasting" - Holanda et al., 2018.

    Args
    ----------
    primary (bool): Whether to model primary production (strongly recommended)
    tau_selection (str): How many tau values to select
        - If 'per-pair', fit tau for each producer-injector pair
        - If 'per-producer', fit tau for each producer (CRMp model)
    constraints (str): How to constrain the gains
        - If 'up-to one' (default), let gains vary from 0 (no connection) to 1 (all injection goes to producer)
        - If 'positive', require each gain to be positive (It is unlikely to go negative in real life)
        - If 'sum-to-one', require the gains for each injector to sum to one
            (all production accounted for)
        - If 'sum-to-one injector', require each injector's gains to sum to one
            (all injection accounted for)

    Examples
    ----------
    forthcoming
    """
    def __init__(self, primary: bool = True,
                 tau_selection: str = 'per-pair',
                 constraints: str = 'positive'):
        if type(primary) != bool:
            raise TypeError('primary must be a boolean')
        self.primary = primary
        if not constraints in ('positive', 'up-to one',
                                 'sum-to-one', 
                                'sum-to-one injector'):
            raise ValueError("Invalid constraints")
        self.constraints = constraints
        self.tau_selection = tau_selection
        if tau_selection == 'per-pair':
            self.q_CRM = q_CRM_perpair
        elif tau_selection == 'per-producer':
            self.q_CRM = q_CRM_perproducer
        else:
            raise ValueError('tau_selection must be one of' +
                             '("per-pair","per-producer")' +
                             f', not {tau_selection}')

    def _get_initial_guess(self, tau_selection: str = '', random=False):
        """Creates the initial guesses for the CRM model parameters

        Args
        ----------
        tau_selection: one of 'per-pair' or 'per-producer',
                       sets whether to use CRM (per-pair) or CRMp model

        Returns
        ----------
        x0 (ndarray): Initial primary production gain, time constant
                      and waterflood gains and time constants, as one long 1-d array
        """
        if tau_selection:
            self.tau_selection = tau_selection

        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        d_t = self.time[1] - self.time[0]
        n_gains, n_tau, n_primary = self._opt_numbers()
        
        if random:
            rng = np.random.default_rng()
            gains_unnormed = rng.random((n_prod, n_inj))
            gains_producer_guess1 = rng.random(n_prod)
        else:
            gains_unnormed = np.ones((n_prod, n_inj))
            gains_producer_guess1 = np.ones(n_prod)
        if self.constraints == 'sum-to-one injector':
            gains_guess1 = gains_unnormed / np.sum(gains_unnormed, 1, keepdims=True)
        else:
            gains_guess1 = gains_unnormed / np.sum(gains_unnormed, 0, keepdims=True)
        tau_producer_guess1 = d_t * np.ones(n_prod)
        if self.tau_selection == 'per-pair':
            tau_guess1 = d_t * np.ones((n_prod, n_inj))
        else:  # 'per-producer'
            tau_guess1 = d_t * np.ones(n_prod)
        if self.primary:
            x0 = np.concatenate([gains_guess1.ravel(), tau_guess1.ravel(),
                                 gains_producer_guess1, tau_producer_guess1
                                ])
        else:
            x0 = np.concatenate([gains_guess1.ravel(), tau_guess1.ravel()])
        return x0

    def _opt_numbers(self):
        """returns the number of gains, taus, and primary production parameters to fit"""
        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        n_gains = n_inj * n_prod
        if self.tau_selection == 'per-pair':
            n_tau = n_gains
        else:
            n_tau = n_prod
        if self.primary:
            n_primary = n_prod * 2
        else:
            n_primary = 0
        return n_gains, n_tau, n_primary
    
    def _get_bounds(self, constraints: str = ''):
        "Create bounds for the model from initialized constraints"
        if constraints:
            self.constraints = constraints
        # some calculations
        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        n_gains, n_tau, n_primary = self._opt_numbers()
        n_all = (n_gains + n_tau + n_primary)
        # setting bounds and constraints
        if self.constraints == 'positive':
            bounds = ((0, np.inf), ) * n_all
            constraints = ()
        elif self.constraints == 'sum-to-one':
            bounds = ((0, np.inf), ) * n_all
            def constrain(x):
                x_gains = (x[:n_gains]
                           .reshape(n_prod, n_inj))
                return sum(np.sum(x_gains, axis=1) - 1)
            constraints = ({'type': 'eq', 'fun': constrain})
        elif self.constraints == 'sum-to-one injector':
            bounds = ((0, np.inf), ) * n_all
            def constrain(x):
                x_gains = (x[:n_gains]
                           .reshape(n_prod, n_inj))
                return np.sum(np.sum(x_gains, axis=0) - 1)
            constraints = ({'type': 'eq', 'fun': constrain})
            #raise NotImplementedError('sum-to-one injector is not implemented')
        elif self.constraints == 'up-to one':
            lb = np.full(n_all, 0)
            ub = np.full(n_all, np.inf)
            ub[:n_gains] = 1
            bounds = tuple(zip(lb, ub))
            constraints = ()
        else:
            raise ValueError('constraints must be one of'
                             '("positive", "sum-to-one","sum-to-one injector","up-to-one"),'
                             f'not {self.constraints}'
                            )
        return bounds, constraints

    def fit(self, production, injection, time, num_cores=1, random=False, **kwargs):
        """Build a CRM model from the production and injection data (production, injection)

        Args
        ----------
        production (ndarray): production rates for each time period, of shape (n_time, n_producers)
        injection (ndarray): injection rates for each time period, of shape (n_time, n_injectors)
        time (ndarray): relative time for each rate measurement, starting from 0, of shape (n_time)
        num_cores (int): number of cores to run fitting procedure on, defaults to 1
        random (bool): whether to randomly initialize the gains
        **kwargs: keyword arguments to pass to scipy.optimize fitting routine

        Returns
        ----------
        self: trained model
        """
        self.production = production
        self.injection = injection
        self.time = time
        n_inj = injection.shape[1]
        if production.shape[0] != injection.shape[0]:
            raise ValueError("production and injection do not have the same number of time steps")
        if production.shape[0] != time.shape[0]:
            raise ValueError("production and time do not have the same number of timesteps")
        
        options = {'maxiter': kwargs.pop('maxiter', 20_000)}
        x0 = self._get_initial_guess(random=random)
        bounds, constraints = self._get_bounds()

        def opts(x):
            n_inj = self.injection.shape[1]
            n_prod = self.production.shape[1]
            n_gains, n_tau, n_primary = self._opt_numbers()
            gains = (x[:n_gains]
                     .reshape(n_prod, n_inj))
            tau = x[n_gains: n_gains + n_tau]
            if len(tau) == n_gains:
                tau = tau.reshape(n_prod, n_inj)
            if self.primary:
                gain_producer = x[n_gains + n_tau: n_gains + n_tau + n_primary // 2]
                tau_producer = x[n_gains + n_tau + n_primary // 2:]
            else:
                gain_producer = np.zeros(n_primary // 2)
                tau_producer = np.ones(n_primary // 2)   
            tau[tau < 1e-10] = 1e-10
            tau_producer[tau_producer < 1e-10] = 1e-10
            return gains, tau, gain_producer, tau_producer

        def calculate_qhat(x, production):
            gains, tau, gain_producer, tau_producer = opts(x)
            n_prod = production.shape[1]
            if self.primary:
                q_hat = q_primary(production, time, gain_producer, tau_producer)
            else:
                q_hat = np.zeros((len(time), n_prod))

            q_hat += self.q_CRM(n_prod, injection, time, gains, tau)
            return q_hat

        def fit_wells(production):
            # residual is an L2 norm
            def residual(x, production):
                return np.sum((production - calculate_qhat(x, production)) ** 2, axis=(0, 1))
           
            result = optimize.minimize(residual, x0, bounds=bounds,
                                       constraints=constraints,
                                       args=(production, ),
                                       options=options,
                                       **kwargs)
            return result
        results = fit_wells(production)

        gains, tau, gains_producer, tau_producer = opts(results['x'])
        self.results = results
        self.gains = gains
        self.tau = tau
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)
        return self

    def predict(self, injection=None, time=None):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection (ndarray): The injection rates to input to the system
        time (ndarray): The timesteps to predict

        Returns
        ----------
        q_hat (ndarray): The predicted values, shape (n_time, n_producers)
        """
        gains, tau, gains_producer, tau_producer = \
            (self.gains, self.tau, self.gains_producer, self.tau_producer)
        production = self.production
        n_producers = production.shape[1]

        if int(injection is None) + int(time is None) == 1:
            raise TypeError("predict() takes 1 or 3 arguments, 2 given")
        if injection is None:
            injection = self.injection
        if time is None:
            time = self.time
        if time.shape[0] != injection.shape[0]:
            raise ValueError("injection and time need the same number of steps")

        q_primary_pred = q_primary(production, time, gains_producer, tau_producer)
        q_secondary = self.q_CRM(n_producers, injection, time, gains, tau)
        q_hat = q_primary_pred + q_secondary
        return q_hat

    def residual(self, production=None, injection=None, time=None):
        """Calculate the production minus the predicted production for a trained model.

        If the production, injection, and time are not provided, this will use the training values

        Args
        ----------
        production (ndarray): The production rates observed, shape (n_timesteps, n_producers)
        injection (ndarray): The injection rates to input to the system, shape (n_timesteps, n_injectors)
        time (ndarray): The timesteps to predict

        Returns
        ----------
        residual (ndarray): The true production data minus the predictions, shape (n_time, n_producers)
        """
        q_hat = self.predict(injection, time)
        if production is None:
            production = self.production
        return production - q_hat

    def to_excel(self, fname):
        "Write trained model to an Excel file"
        for x in ('gains', 'tau', 'gains_producer', 'tau_producer'):
            if x not in self.__dict__.keys():
                raise(ValueError('Model has not been trained'))
        with pd.ExcelWriter(fname) as f:
            pd.DataFrame(self.gains).to_excel(f, sheet_name='Gains')
            pd.DataFrame(self.tau).to_excel(f, sheet_name='Taus')
            (pd.DataFrame({'Producer gains': self.gains_producer,
                           'Producer taus': self.tau_producer}
                          )
             .to_excel(f, sheet_name='Primary production')
             )

    def to_pickle(self, fname):
        "Write trained model to a pickle file"
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
