import numpy as np
from numba import jit, njit, prange
import pandas as pd
import pickle
from scipy import optimize
from joblib import Parallel, delayed


@njit
def q_primary(production, time, gain_producer, tau_producer):
    """Calculates primary production contribution using Arps equation with b=0

    Args
    ----------
    production (ndarray): Production, size: Number of time steps
    time (ndarray): Producing times to forecast, size: Number of time steps
    gain_producer: Arps q_i factor
    tau_producer: Arps time constant

    Returns
    ----------
    q_hat: Calculated production, size: Number of time steps
    """
    q_hat = np.zeros_like(production)

    # Compute primary production component
    for k in range(len(time)):
        time_decay = np.exp(- time[k] / tau_producer)
        q_hat[k] += gain_producer * production[0] * time_decay
    return q_hat


@njit
def q_CRM_perpair(injection, time, gains, taus):
    """Calculates per injector-producer pair production for all injectors on one producer
    using CRM model

    Args
    ----------
    injection (ndarray): Production, size: Number of time steps
    time (ndarray): Producing times to forecast, size: Number of time steps
    gains (ndarray): Connectivities between each injector and the producer, size: Number of injectors
    taus (ndarray): Time constants between each injector and the producer, size: Number of injectors

    Returns
    ----------
    q_hat: Calculated production, size: Number of time steps
    """
    n = len(time)
    q_hat = np.zeros(n)
    conv_injected = np.zeros((n, injection.shape[1]))

    # Compute convolved injection rates
    for j in range(injection.shape[1]):
        conv_injected[0, j] += ((1 - np.exp((time[0] - time[1]) / taus[j]))
                                * injection[0, j])
        for k in range(1, n):
            for l in range(1, k+1):
                time_decay = ((1 - np.exp((time[l-1] - time[l]) / taus[j]))
                              * np.exp((time[l] - time[k]) / taus[j])
                              )
                conv_injected[k, j] += time_decay * injection[l, j]

    # Calculate waterflood rates
    for k in range(n):
        for j in range(injection.shape[1]):
            q_hat[k] += gains[j] * conv_injected[k, j]
    return q_hat


@njit
def q_CRM_perproducer(injection, time, gain, tau):
    """Calculates per injector-producer pair production for all injectors on one producer
    using simplified CRMp model that assumes a single tau for each producer

    Args
    ----------
    injection (ndarray): Production, size: Number of time steps
    time (ndarray): Producing times to forecast, size: Number of time steps
    gains (ndarray): Connectivities between each injector and the producer, size: Number of injectors
    tau: Time constants all injectors and the producer

    Returns
    ----------
    q_hat: Calculated production, size: Number of time steps
    """
    tau2 = tau * np.ones(injection.shape[1])
    return q_CRM_perpair(injection, time, gain, tau2)


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
        - If 'sum-to-one injector' (not implemented), require each injector's gains to sum to one
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

    def _get_initial_guess(self, tau_selection: str = ''):
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
        d_t = self.time[1] - self.time[0]
        gains_guess1 = np.ones(n_inj) / n_inj
        gains_producer_guess1 = 1
        tau_producer_guess1 = d_t
        if self.tau_selection == 'per-pair':
            tau_guess1 = np.ones(n_inj) * d_t
        else:  # 'per-producer'
            tau_guess1 = np.array([d_t])
        if self.primary:
            x0 = np.concatenate([gains_guess1, tau_guess1,
                                 [gains_producer_guess1, tau_producer_guess1]])
        else:
            x0 = np.concatenate([gains_guess1, tau_guess1])
        return x0

    def _get_bounds(self, constraints: str = ''):
        "Create bounds for the model from initialized constraints"
        if constraints:
            self.constraints = constraints

        n_inj = self.injection.shape[1]
        if self.tau_selection == 'per-pair':
            n = n_inj * 2
        else:
            n = n_inj + 1

        if self.primary:
            n = n + 2

        if self.constraints == 'positive':
            bounds = ((0, np.inf), ) * n
            constraints = ()
        elif self.constraints == 'sum-to-one':
            bounds = ((0, np.inf), ) * n

            def constrain(x):
                x = x[:n_inj]
                return np.sum(x) - 1
            constraints = ({'type': 'eq', 'fun': constrain})
        elif self.constraints == 'sum-to-one injector':
            raise NotImplementedError('sum-to-one injector is not implemented')
        elif self.constraints == 'up-to one':
            lb = np.full(n, 0)
            ub = np.full(n, np.inf)
            ub[:n_inj] = 1
            bounds = tuple(zip(lb, ub))
            constraints = ()
        else:
            bounds = ((0, np.inf), ) * n
            constraints = ()
        return bounds, constraints

    def fit(self, production, injection, time, num_cores=1, **kwargs):
        """Build a CRM model from the production and injection data (production, injection)

        Args
        ----------
        production (ndarray): production rates for each time period, of shape (n_time, n_producers)
        injection (ndarray): injection rates for each time period, of shape (n_time, n_injectors)
        time (ndarray): relative time for each rate measurement, starting from 0, of shape (n_time)
        num_cores (int): number of cores to run fitting procedure on, defaults to 1
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

        x0 = self._get_initial_guess()
        bounds, constraints = self._get_bounds()

        def opts(x):
            gains = x[:n_inj]
            if self.tau_selection == 'per-pair':
                tau = x[n_inj:n_inj * 2]
            else:
                tau = x[n_inj]
            if self.primary:
                gain_producer = x[-2]
                tau_producer = x[-1]
            else:
                gain_producer = 0
                tau_producer = 1
            if self.tau_selection == 'per-pair':
                tau[tau < 1e-10] = 1e-10
            elif tau < 1e-10:
                tau = 1e-10
            if tau_producer < 1e-10:
                tau_producer = 1e-10
            return gains, tau, gain_producer, tau_producer

        def calculate_qhat(x, production):
            gains, tau, gain_producer, tau_producer = opts(x)
            if self.primary:
                q_hat = q_primary(production, time, gain_producer, tau_producer)
            else:
                q_hat = np.zeros(len(time))

            q_hat += self.q_CRM(injection, time, gains, tau)
            return q_hat

        def fit_well(production):
            # residual is an L2 norm
            def residual(x, production):
                return sum((production - calculate_qhat(x, production)) ** 2)

            result = optimize.minimize(residual, x0, bounds=bounds,
                                       constraints=constraints,
                                       args=(production, ),
                                       **kwargs)
            return result

        production_perwell = [x for x in self.production.T]
        if num_cores == 1:
            results = map(fit_well, production_perwell)
        else:
            results = Parallel(n_jobs=num_cores)(delayed(fit_well)(x) for x in self.production.T)

        opts_perwell = [opts(r['x']) for r in results]
        gains_perwell, tau_perwell, gains_producer, tau_producer = \
            map(list, zip(*opts_perwell))

        self.gains = np.vstack(gains_perwell)
        self.tau = np.vstack(tau_perwell)
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

        if injection is None:
            injection = self.injection
        if time is None:
            time = self.time

        q_hat = np.zeros((len(time), n_producers))
        for i in range(n_producers):
            q_hat[:, i] += q_primary(production[:, i], time, gains_producer[i],
                                     tau_producer[i])
            q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
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
