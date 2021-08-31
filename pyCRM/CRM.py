import numpy as np
import jax.numpy as jnp
from jax import jit
import pandas as pd
import pickle

# from jax.scipy import optimize
from scipy import optimize

# from joblib import Parallel, delayed


@jit
def q_primary(production, time, gain_producer, tau_producer):
    """Calculates primary production contribution using Arps equation with b=0

    Args
    ----------
    production (ndarray): Production, size: Number of time steps x producers
    time (ndarray): Producing times to forecast, size: Number of time steps x producers
    gain_producer (ndarray): Arps q_i factor, size: Number of producers
    tau_producer (ndarray): Arps time constant, size: Number of producers

    Returns
    ----------
    q_hat (ndarray): Calculated production, size: Number of time steps x producers
    """
    decay = jnp.exp(-jnp.einsum("i,j", time, 1 / tau_producer))
    q_hat = decay * production[0, :] * gain_producer
    return q_hat


@jit
def calc_time_decay(time, taus):
    time2 = jnp.atleast_2d(time)
    time_diff = jnp.apply_along_axis(lambda x: x - time, axis=0, arr=time2)
    time_diff = jnp.where(time_diff >= 0, time_diff, jnp.inf)
    time_diff_scaled = jnp.outer(time_diff, 1 / taus).reshape(
        (len(time), len(time), *taus.shape)
    )
    time_decay = jnp.exp(-time_diff_scaled)
    return time_decay


@jit
def calc_nearest_time_decay(time, taus):
    time_diff = jnp.concatenate([jnp.array([time[1] - time[0]]), time[1:] - time[:-1]])
    time_diff_scaled = jnp.outer(time_diff, 1 / taus).reshape(len(time), *taus.shape)
    time_decay = jnp.exp(-time_diff_scaled)
    return time_decay


@jit
def q_CRM_perproducer(injection, time, gains, tau):
    tau2 = jnp.einsum("i,ij->ij", tau, jnp.ones_like(gains))
    return q_CRM_perpair(injection, time, gains, tau2)


@jit
def q_CRM_perpair(injection, time, gains, tau):
    """Calculates per injector-producer pair production for all injectors on one producer
    using CRM model

    Args
    ----------
    injection (ndarray): Injected fluid
    time (ndarray): Producing times to forecast
    gains (ndarray): Connectivities between each injector and the producer
    taus (ndarray): Time constants between each injector and the producer

    Returns
    ----------
    q_hat: Calculated production
    """
    # i=injector, p=producer, k=production time, l=injection time
    time_decay = calc_time_decay(time, tau)
    neighbor_time_decay = calc_nearest_time_decay(time, tau)
    gained_injection = jnp.einsum("li,pi->lip", injection, gains)
    total_decay = jnp.einsum("kpi,lkpi->klip", (1 - neighbor_time_decay), time_decay)
    q_hat = jnp.einsum("klip,lip->kp", total_decay, gained_injection)
    return q_hat


def random_weights(n_i: int, n_j: int, axis: int = 0, seed=None):
    rng = np.random.default_rng(seed)
    limit = 10 * (n_i if axis == 0 else n_j)
    vec = rng.integers(0, limit, (n_i, n_j))
    axis_sum = vec.sum(axis, keepdims=True)
    return vec / axis_sum


class CRM:
    """A Capacitance Resistance Model history matcher

    CRM uses a physics-inspired mass balance approach to explain production for
    waterfloods. It treats each injector-producer well pair as a system with mass input,
    output, and pressure related to the mass balance. Several versions exist. For an
    exhaustive review, check "A State-of-the-Art Literature Review on Capacitance
    Resistance Models for Reservoir Characterization and Performance Forecasting" -
    Holanda et al., 2018.

    Args
    ----------
    primary (bool): Whether to model primary production (strongly recommended)
    tau_selection (str): How many tau values to select
        - If 'per-pair', fit tau for each producer-injector pair
        - If 'per-producer', fit tau for each producer (CRMp model)
    constraints (str): How to constrain the gains
        - If 'up-to one' (default), let gains vary from 0 (no connection) to
            1 (all injection goes to producer)
        - If 'positive', require each gain to be positive
            (It is unlikely to go negative in real life)
        - If 'sum-to-one', require the gains for each injector to sum to one
            (all production accounted for)
        - If 'sum-to-one injector', require each injector's gains to sum to one
            (all injection accounted for)

    Examples
    ----------
    forthcoming
    """

    def __init__(
        self,
        primary: bool = True,
        tau_selection: str = "per-pair",
        constraints: str = "positive",
    ):
        if type(primary) != bool:
            raise TypeError("primary must be a boolean")
        self.primary = primary
        if constraints not in (
            "positive",
            "up-to one",
            "sum-to-one",
            "sum-to-one injector",
        ):
            raise ValueError("Invalid constraints")
        self.constraints = constraints
        self.tau_selection = tau_selection
        if tau_selection == "per-pair":
            self.q_CRM = q_CRM_perpair
        elif tau_selection == "per-producer":
            self.q_CRM = q_CRM_perproducer
        else:
            raise ValueError(
                "tau_selection must be one of"
                + '("per-pair","per-producer")'
                + f", not {tau_selection}"
            )

    def _get_initial_guess(self, tau_selection: str = "", random=False):
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

        axis = 1 if (self.constraints == "sum-to-one injector") else 0
        if random:
            rng = np.random.default_rng()
            gains_producer_guess1 = rng.random(n_prod)
            gains_guess1 = random_weights(n_prod, n_inj, axis)
        else:
            gains_unnormed = np.ones((n_prod, n_inj))
            gains_guess1 = gains_unnormed / np.sum(gains_unnormed, axis, keepdims=True)
            gains_producer_guess1 = np.ones(n_prod)
        tau_producer_guess1 = d_t * np.ones(n_prod)
        if self.tau_selection == "per-pair":
            tau_guess1 = d_t * np.ones((n_prod, n_inj))
        else:  # 'per-producer'
            tau_guess1 = d_t * np.ones(n_prod)
        if self.primary:
            x0 = np.concatenate(
                [
                    gains_guess1.ravel(),
                    tau_guess1.ravel(),
                    gains_producer_guess1,
                    tau_producer_guess1,
                ]
            )
        else:
            x0 = np.concatenate([gains_guess1.ravel(), tau_guess1.ravel()])
        return x0

    def _opt_numbers(self):
        """
        returns the number of gains, taus,
        and primary production parameters to fit
        """
        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        n_gains = n_inj * n_prod
        if self.tau_selection == "per-pair":
            n_tau = n_gains
        else:
            n_tau = n_prod
        if self.primary:
            n_primary = n_prod * 2
        else:
            n_primary = 0
        return n_gains, n_tau, n_primary

    def _get_bounds(self, constraints: str = ""):
        "Create bounds for the model from initialized constraints"
        if constraints:
            self.constraints = constraints
        # some calculations
        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        n_gains, n_tau, n_primary = self._opt_numbers()

        # setting bounds and constraints
        tau_max = (self.time[-1] - self.time[0]) * 5
        tau_min = (self.time[1] - self.time[0]) / 3
        tau_bounds = [(tau_min, tau_max)]
        if self.constraints == "positive":
            bounds = (
                [(0, np.inf)] * n_gains + tau_bounds * n_tau + [(0, np.inf)] * n_primary
            )
            constraints = ()
        elif self.constraints == "sum-to-one":
            bounds = (
                [(0, 1)] * n_gains
                + tau_bounds * n_tau
                + [(0, 1)] * (n_primary // 2)
                + tau_bounds * (n_primary // 2)
            )

            def constrain(x):
                x_gains = x[:n_gains].reshape(n_prod, n_inj)
                return np.max(np.sum(x_gains, axis=1) - 1)

            constraints = {"type": "eq", "fun": constrain}
        elif self.constraints == "sum-to-one injector":
            bounds = (
                [(0, 1)] * n_gains
                + tau_bounds * n_tau
                + [(0, 1)] * (n_primary // 2)
                + tau_bounds * (n_primary // 2)
            )

            def constrain(x):
                x_gains = x[:n_gains].reshape(n_prod, n_inj)
                return np.max(np.sum(x_gains, axis=0) - 1)

            constraints = {"type": "ineq", "fun": constrain}
            # raise NotImplementedError('sum-to-one injector is not implemented')
        elif self.constraints == "up-to one":
            bounds = (
                [(0, 1)] * n_gains
                + tau_bounds * n_tau
                + [(0, 1)] * (n_primary // 2)
                + tau_bounds * (n_primary // 2)
            )
            constraints = ()
        else:
            raise ValueError(
                "constraints must be one of"
                '("positive", "sum-to-one","sum-to-one injector","up-to-one"),'
                f"not {self.constraints}"
            )
        return bounds, constraints

    def _split_opts(self, x: np.ndarray):
        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        n_gains, n_tau, n_primary = self._opt_numbers()
        gains = x[:n_gains].reshape(n_prod, n_inj)
        tau = x[n_gains : n_gains + n_tau]
        if len(tau) == n_gains:
            tau = tau.reshape(n_prod, n_inj)
        if self.primary:
            gain_producer = x[n_gains + n_tau : n_gains + n_tau + n_primary // 2]
            tau_producer = x[n_gains + n_tau + n_primary // 2 :]
        else:
            gain_producer = jnp.zeros(n_primary // 2)
            tau_producer = jnp.ones(n_primary // 2)
        tau = jnp.where(tau < 1e-10, 1e-10, tau)
        tau_producer = jnp.where(tau_producer < 1e-10, 1e-10, tau_producer)
        return gains, tau, gain_producer, tau_producer

    def _calculate_qhat(self, x, production, injection, time):
        gains, tau, gain_producer, tau_producer = self._split_opts(x)
        n_prod = production.shape[1]
        if self.primary:
            q_hat = q_primary(production, time, gain_producer, tau_producer)
        else:
            q_hat = jnp.zeros((len(time), n_prod))

        q_hat += self.q_CRM(injection, time, gains, tau)
        return q_hat

    def fit(
        self,
        production,
        injection,
        time,
        initial_guess=None,
        random=False,
        global_fit=False,
        **kwargs,
    ):
        """Build a CRM model from the production and injection data

        Args
        ----------
        production (ndarray): production rates for each time period, of shape
            (n_time, n_producers)
        injection (ndarray): injection rates for each time period, of shape
            (n_time, n_injectors)
        time (ndarray): relative time for each rate measurement, starting from 0, of
            shape (n_time)
        initial_guess (ndarray): initial guesses for gains, taus, primary production
            contribution, as one-dimensional ndarray
        random (bool): whether to randomly initialize the gains
        global (bool): whether to use a global optimizer
        **kwargs: keyword arguments to pass to scipy.optimize fitting routine
            default method is `trust-constr`

        Returns
        ----------
        self: trained model
        """
        self.production = production
        self.injection = injection
        self.time = time
        if production.shape[0] != injection.shape[0]:
            raise ValueError(
                "production and injection do not have the same number of time steps"
            )
        if production.shape[0] != time.shape[0]:
            raise ValueError(
                "production and time do not have the same number of timesteps"
            )
        if initial_guess is None:
            initial_guess = self._get_initial_guess(random=random)
        bounds, constraints = self._get_bounds()
        if "method" not in kwargs and not global_fit:
            kwargs["method"] = "trust-constr"

        def fit_wells(production, injection, time):
            production = jnp.array(production)
            injection = jnp.array(injection)
            time = jnp.array(time)

            def residual(x, production, injection, time):
                "residual is an L2 norm"
                err = production - self._calculate_qhat(x, production, injection, time)
                return jnp.sum(err ** 2, axis=(0, 1))

            if global_fit:
                result = optimize.shgo(
                    residual,
                    bounds=bounds,
                    constraints=constraints,
                    args=(production, injection, time),
                    **kwargs,
                )
            else:
                result = optimize.minimize(
                    residual,
                    initial_guess,
                    bounds=bounds,
                    constraints=constraints,
                    args=(production, injection, time),
                    **kwargs,
                )
            return result

        results = fit_wells(production, injection, time)

        gains, tau, gains_producer, tau_producer = self._split_opts(results.x)
        self.results = results
        self.set_connections(
            gains, tau, np.array(gains_producer), np.array(tau_producer)
        )
        return self

    def predict(self, injection=None, time=None, connections={}):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection (ndarray): The injection rates to input to the system
        time (ndarray): The timesteps to predict
        connections (dict): if present, the gains, tau, gains_producer, tau_producer
            matrices

        Returns
        ----------
        q_hat (ndarray): The predicted values, shape (n_time, n_producers)
        """
        gains = connections["gains"] if "gains" in connections else self.gains
        tau = connections["tau"] if "tau" in connections else self.tau
        gains_producer = (
            connections["gains_producer"]
            if "gains_producer" in connections
            else self.gains_producer
        )
        tau_producer = (
            connections["tau_producer"]
            if "tau_producer" in connections
            else self.tau_producer
        )
        production = self.production

        if int(injection is None) + int(time is None) == 1:
            raise TypeError("predict() takes 1 or 3 arguments, 2 given")
        if injection is None:
            injection = self.injection
        if time is None:
            time = self.time
        if time.shape[0] != injection.shape[0]:
            raise ValueError("injection and time need the same number of steps")

        q_primary_pred = q_primary(production, time, gains_producer, tau_producer)
        q_secondary = self.q_CRM(injection, time, gains, tau)
        q_hat = q_primary_pred + q_secondary
        return q_hat

    def set_rates(self, production=None, injection=None, time=None):
        """Sets production and injection rates and time"""
        if production is not None:
            self.production = production
        if injection is not None:
            self.injection = injection
        if time is not None:
            self.time = time

    def set_connections(
        self, gains=None, tau=None, gains_producer=None, tau_producer=None
    ):
        """Sets waterflood properties"""
        if gains is not None:
            self.gains = gains
        if tau is not None:
            self.tau = tau
        if gains_producer is not None:
            self.gains_producer = gains_producer
        if tau_producer is not None:
            self.tau_producer = tau_producer

    def residual(self, production=None, injection=None, time=None, connections={}):
        """Calculate the production minus the predicted production for a trained model.

        If the production, injection, and time are not provided, this will use the
        training values

        Args
        ----------
        production (ndarray): The production rates observed, shape (n_timesteps,
            n_producers)
        injection (ndarray): The injection rates to input to the system, shape
            (n_timesteps, n_injectors)
        time (ndarray): The timesteps to predict
        connections (dict): gains, tau, gains_producer, tau_producer arguments

        Returns
        ----------
        residual (ndarray): The true production data minus the predictions, shape
            (n_time, n_producers)
        """
        q_hat = self.predict(injection, time, connections=connections)
        if production is None:
            production = self.production
        return production - q_hat

    def to_excel(self, fname):
        "Write trained model to an Excel file"
        for x in ("gains", "tau", "gains_producer", "tau_producer"):
            if x not in self.__dict__.keys():
                raise (ValueError("Model has not been trained"))
        with pd.ExcelWriter(fname) as f:
            pd.DataFrame(self.gains).to_excel(f, sheet_name="Gains")
            pd.DataFrame(self.tau).to_excel(f, sheet_name="Taus")
            (
                pd.DataFrame(
                    {
                        "Producer gains": self.gains_producer,
                        "Producer taus": self.tau_producer,
                    }
                ).to_excel(f, sheet_name="Primary production")
            )

    def to_pickle(self, fname):
        "Write trained model to a pickle file"
        with open(fname, "wb") as f:
            pickle.dump(self, f)
