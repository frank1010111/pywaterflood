import numpy as np
from numba import jit, njit, prange
import pandas as pd
import pickle
from scipy import optimize


@njit
def q_primary(production, time, gain_producer, tau_producer):
    q_hat = np.zeros_like(production)

    # Compute primary production component
    for k in range(len(time)):
        time_decay = np.exp(- time[k] / tau_producer)
        q_hat[k] += gain_producer * production[0] * time_decay
    return q_hat


@njit
def q_CRM_perpair(injection, time, gains, taus):
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
    tau2 = tau * np.ones(injection.shape[1])
    return q_CRM_perpair(injection, time, gain, tau2)


class CRM():
    def __init__(self, primary: bool = True,
                 tau_selection: str = 'per-pair',
                 constraints: str = 'positive'):
        self.primary = primary
        assert ((constraints in ('positive', 'up-to one',
                                 'sum-to-one', 'sum-to-one injector')),
                "Invalid constraints")
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

    def get_initial_guess(self, tau_selection: str = ''):
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
        x0 = np.concatenate([gains_guess1, tau_guess1,
                             [gains_producer_guess1, tau_producer_guess1]])
        return x0

    def get_bounds(self, constraints: str = ''):
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
        self.production = production
        self.injection = injection
        self.time = time
        n_inj = injection.shape[1]

        x0 = self.get_initial_guess()
        bounds, constraints = self.get_bounds()

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
            tau[tau < 1e-10] = 1e-10
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
            raise NotImplementedError(
                "Multi-core optimization has not yet been implemented," +
                " please set num_cores=1")

        opts_perwell = [opts(r['x']) for r in results]
        gains_perwell, tau_perwell, gains_producer, tau_producer = \
            map(list, zip(*opts_perwell))

        self.gains = np.vstack(gains_perwell)
        self.tau = np.vstack(tau_perwell)
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)

    def predict(self, injection=None, time=None):
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
        q_hat = self.predict(injection, time)
        if production is None:
            production = self.production
        return production - q_hat

    def to_excel(self, fname):
        with pd.ExcelWriter(fname) as f:
            pd.DataFrame(self.gains).to_excel(f, sheet_name='Gains')
            pd.DataFrame(self.tau).to_excel(f, sheet_name='Taus')
            (pd.DataFrame({'Producer gains': self.gains_producer,
                           'Producer taus': self.tau_producer}
                          )
             .to_excel(f, sheet_name='Primary production')
             )

    def to_pickle(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
