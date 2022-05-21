"""Analyze waterfloods with capacitance-resistance models. # noqa: D401,D400

Classes
-------
CRM : standard capacitance resistance modeling
CrmCompensated : including pressure

Methods
-------
q_primary : primary production
q_CRM_perpair : production due to injection (injector-producer pairs)
q_CRM_perproducer : production due to injection (one producer, many injectors)
q_bhp : production from changing bottomhole pressures of producers
"""
from __future__ import annotations

import pickle
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from numpy import ndarray
from scipy import optimize


@njit
def q_primary(
    production: ndarray, time: ndarray, gain_producer: ndarray, tau_producer: ndarray
) -> ndarray:
    """Calculate primary production contribution.

    Uses Arps equation with b=0
    .. math::
        q_{p}(t) = q_i e^{-bt}

    Args
    ----------
    production : ndarray
        Production, size: Number of time steps
    time : ndarray
        Producing times to forecast, size: Number of time steps
    gain_producer : ndarray
        Arps q_i factor
    tau_producer : ndarray
        Arps time constant

    Returns
    ----------
    q_hat : ndarray
        Calculated production, size: Number of time steps
    """
    time_decay = np.exp(-time / tau_producer)
    q_hat = time_decay * production[0] * gain_producer
    return q_hat


@njit
def q_CRM_perpair(injection: ndarray, time: ndarray, gains: ndarray, taus: ndarray) -> ndarray:
    """Calculate per injector-producer pair production.

    Runs for influences of each injector on one producer, assuming
    individual `gain`s and `tau`s for each pair

    Args
    ----------
    injection : ndarray
        Injected fluid, size: Number of time steps
    time : ndarray
        Producing times to forecast, size: Number of time steps
    gains : ndarray
        Connectivities between each injector and the producer,
        size: Number of injectors
    taus : ndarray
        Time constants between each injector and the producer,
        size: Number of injectors

    Returns
    ----------
    q_hat : ndarray
        Calculated production, size: Number of time steps
    """
    n = len(time)
    q_hat = np.zeros(n)
    conv_injected = np.zeros((n, injection.shape[1]))

    # Compute convolved injection rates
    for j in range(injection.shape[1]):
        conv_injected[0, j] += (1 - np.exp((time[0] - time[1]) / taus[j])) * injection[0, j]
        for k in range(1, n):
            for m in range(1, k + 1):
                time_decay = (1 - np.exp((time[m - 1] - time[m]) / taus[j])) * np.exp(
                    (time[m] - time[k]) / taus[j]
                )
                conv_injected[k, j] += time_decay * injection[m, j]

    # Calculate waterflood rates
    for k in range(n):
        for j in range(injection.shape[1]):
            q_hat[k] += gains[j] * conv_injected[k, j]
    return q_hat


@njit
def q_CRM_perproducer(injection: ndarray, time: ndarray, gain: ndarray, tau: float) -> ndarray:
    """Calculate per injector-producer pair production (simplified tank).

    Uses simplified CRMp model that assumes a single tau for each producer

    Args
    ----------
    injection : ndarray
        injected fluid in reservoir volumes, size: Number of time steps
    time : ndarray
        Producing times to forecast, size: Number of time steps
    gains : ndarray
        Connectivities between each injector and the producer
        size: Number of injectors
    tau : float
        Time constants all injectors and the producer

    Returns
    ----------
    q_hat : ndarray
        Calculated production, size: Number of time steps
    """
    tau2 = tau * np.ones(injection.shape[1])
    return q_CRM_perpair(injection, time, gain, tau2)


@njit
def _pressure_diff(pressure_local: ndarray, pressure: ndarray) -> ndarray:
    """Pressure differences from local to each producer each timestep."""
    n_t, n_p = pressure.shape
    pressure_diff = np.zeros((n_p, n_t))
    for j in range(n_p):
        for t in range(1, n_t):
            pressure_diff[j, t] = pressure_local[t - 1] - pressure[t, j]
    return pressure_diff


def q_bhp(pressure_local: ndarray, pressure: ndarray, v_matrix: ndarray) -> ndarray:
    r"""Calculate the production effect from bottom-hole pressure variation.

    This looks like
    .. math::
        q_{BHP,j}(t_i) = \sum_{k} v_{kj}\left[ p_j(t_{i-1}) - p_k(t_i) \right]

    Args
    ----
    pressure_local : ndarray
        pressure for the well in question, shape: n_time
    pressure : ndarray
        bottomhole pressure, shape: n_time, n_producers
    v_matrix : ndarray
        connectivity between one producer and all producers, shape: n_producers

    Returns
    -------
    q : ndarray
        production from changing BHP
        shape: n_time
    """
    pressure_diff = _pressure_diff(pressure_local, pressure)
    q = np.einsum("j,jt->t", v_matrix, pressure_diff)
    return q


def random_weights(n_i: int, n_j: int, axis: int = 0, seed: int | None = None) -> ndarray:
    """Generate random weights for producer-injector gains.

    Args
    ----
    n_i : int
    n_j : int
    axis : int, default is 0
    seed : int, default is None

    Returns
    -------
    gains_guess: ndarray
    """
    rng = np.random.default_rng(seed)
    limit = 10 * (n_i if axis == 0 else n_j)
    vec = rng.integers(0, limit, (n_i, n_j))
    axis_sum = vec.sum(axis, keepdims=True)
    return vec / axis_sum


class CRM:
    """A Capacitance Resistance Model history matcher.

    CRM uses a physics-inspired mass balance approach to explain production for \
        waterfloods. It treats each injector-producer well pair as a system \
        with mass input, output, and pressure related to the mass balance. \
        Several versions exist. Select them from the arguments.

    Args
    ----------
    primary : bool
        Whether to model primary production (strongly recommended)
    tau_selection : str
        How many tau values to select
            - If 'per-pair', fit tau for each producer-injector pair
            - If 'per-producer', fit tau for each producer (CRMp model)
    constraints : str
        How to constrain the gains
            * If 'up-to one' (default), let gains vary from 0 (no connection) to 1 \
           (all injection goes to producer)
            * If 'positive', require each gain to be positive \
                (It is unlikely to go negative in real life)
            * If 'sum-to-one', require the gains for each injector to sum to one \
                            (all production accounted for)
            * If 'sum-to-one injector' (not implemented), require each injector's \
                gains to sum to one (all injection accounted for)

    Examples
    ----------
    crm = CRM(True, "per-pair", "up-to one")

    References
    ----------
    "A State-of-the-Art Literature Review on Capacitance Resistance Models for
    Reservoir Characterization and Performance Forecasting" - Holanda et al., 2018.
    """

    def __init__(
        self,
        primary: bool = True,
        tau_selection: str = "per-pair",
        constraints: str = "positive",
    ):
        """Initialize CRM with appropriate settings."""
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

    def fit(
        self,
        production: ndarray,
        injection: ndarray,
        time: ndarray,
        initial_guess: ndarray = None,
        num_cores: int = 1,
        random: bool = False,
        **kwargs,
    ):
        """Build a CRM model from the production and injection data.

        Args
        ----------
        production : ndarray
            production rates for each time period,
            shape: (n_time, n_producers)
        injection : ndarray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        time : ndarray
            relative time for each rate measurement, starting from 0,
            shape: (n_time)
        initial_guess : ndarray
            initial guesses for gains, taus, primary production contribution
            shape: (len(guess), n_producers)
        num_cores (int): number of cores to run fitting procedure on, defaults to 1
        random : bool
            whether to randomly initialize the gains
        **kwargs:
            keyword arguments to pass to scipy.optimize fitting routine

        Returns
        ----------
        self: trained model
        """
        _validate_inputs(production, injection, time)
        self.production = production
        self.injection = injection
        self.time = time

        if not initial_guess:
            initial_guess = self._get_initial_guess(random=random)
        bounds, constraints = self._get_bounds()
        num_cores = kwargs.pop("num_cores", 1)

        def fit_well(production, x0):
            # residual is an L2 norm
            def residual(x, production):
                return sum(
                    (production - self._calculate_qhat(x, production, injection, time)) ** 2
                )

            result = optimize.minimize(
                residual,
                x0,
                bounds=bounds,
                constraints=constraints,
                args=(production,),
                **kwargs,
            )
            return result

        if num_cores == 1:
            results = map(fit_well, self.production.T, initial_guess)
        else:
            results = Parallel(n_jobs=num_cores)(
                delayed(fit_well)(p, x0) for p, x0 in zip(self.production.T, initial_guess)
            )

        opts_perwell = [self._split_opts(r["x"]) for r in results]
        gains_perwell, tau_perwell, gains_producer, tau_producer = map(list, zip(*opts_perwell))

        self.gains = np.vstack(gains_perwell)
        self.tau = np.vstack(tau_perwell)
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)
        return self

    def predict(self, injection=None, time=None, connections=None):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection : ndarray
            The injection rates to input to the system, shape (n_time, n_inj)
        time : ndarray
            The timesteps to predict
        connections : dict
            if present, the gains, tau, gains_producer, tau_producer
            matrices

        Returns
        ----------
        q_hat :ndarray
            The predicted values, shape (n_time, n_producers)
        """
        if connections is not None:
            gains = connections.get("gains", self.gains)
            tau = connections.get("tau", self.tau)
            gains_producer = connections.get("gains_producer", self.gains_producer)
            tau_producer = connections.get("tau_producer", self.tau_producer)
        else:
            gains = self.gains
            tau = self.tau
            gains_producer = self.gains_producer
            tau_producer = self.tau_producer
        production = self.production
        n_producers = production.shape[1]

        if int(injection is None) + int(time is None) == 1:
            raise TypeError("predict() takes 1 or 3 arguments, 2 given")
        if injection is None:
            injection = self.injection
        if time is None:
            time = self.time
        if time.shape[0] != injection.shape[0]:
            raise ValueError("injection and time need same number of steps")

        q_hat = np.zeros((len(time), n_producers))
        for i in range(n_producers):
            q_hat[:, i] += q_primary(production[:, i], time, gains_producer[i], tau_producer[i])
            q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
        return q_hat

    def set_rates(self, production=None, injection=None, time=None):
        """Set production and injection rates and time array.

        Args
        -----
        production : ndarray
            production rates with shape (n_time, n_producers)
        injection : ndarray
            injection rates with shape (n_time, n_injectors)
        time : ndarray
            timesteps with shape n_time
        """
        _validate_inputs(production, injection, time)
        if production is not None:
            self.production = production
        if injection is not None:
            self.injection = injection
        if time is not None:
            self.time = time

    def set_connections(self, gains=None, tau=None, gains_producer=None, tau_producer=None):
        """Set waterflood properties.

        Args
        -----
        gains : ndarray
            connectivity between injector and producer
            shape: n_gains, n_producers
        tau : ndarray
            time-constant for injection to be felt by production
            shape: either n_producers or (n_gains, n_producers)
        gains_producer : ndarray
            gain on primary production, shape: n_producers
        tau_producer : ndarray
            Arps time constant for primary production, shape: n_producers
        """
        if gains is not None:
            self.gains = gains
        if tau is not None:
            self.tau = tau
        if gains_producer is not None:
            self.gains_producer = gains_producer
        if tau_producer is not None:
            self.tau_producer = tau_producer

    def residual(self, production=None, injection=None, time=None):
        """Calculate the production minus the predicted production for a trained model.

        If the production, injection, and time are not provided, this will use the
         training values

        Args
        ----------
        production : ndarray
            The production rates observed, shape: (n_timesteps, n_producers)
        injection : ndarray
            The injection rates to input to the system,
            shape: (n_timesteps, n_injectors)
        time : ndarray
            The timesteps to predict

        Returns
        ----------
        residual : ndarray
            The true production data minus the predictions, shape (n_time, n_producers)
        """
        q_hat = self.predict(injection, time)
        if production is None:
            production = self.production
        return production - q_hat

    def to_excel(self, fname: str):
        """Write trained model to an Excel file.

        Args
        ----
        fname : str
            Excel file to write out

        """
        for x in ("gains", "tau", "gains_producer", "tau_producer"):
            if x not in self.__dict__.keys():
                raise (ValueError("Model has not been trained"))
        with pd.ExcelWriter(fname) as f:
            pd.DataFrame(self.gains).to_excel(f, sheet_name="Gains")
            pd.DataFrame(self.tau).to_excel(f, sheet_name="Taus")
            pd.DataFrame(
                {
                    "Producer gains": self.gains_producer,
                    "Producer taus": self.tau_producer,
                }
            ).to_excel(f, sheet_name="Primary production")

    def to_pickle(self, fname: str):
        """Write trained model to a pickle file.

        Args
        -----
        fname : str
            pickle file to write out
        """
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def _get_initial_guess(self, tau_selection: str | None = None, random=False):
        """Create initial guesses for the CRM model parameters.

        :meta private:

        Args
        ----------
        tau_selection : str, one of 'per-pair' or 'per-producer'
            sets whether to use CRM (per-pair) or CRMp model
        random : bool
            whether initial gains are randomly (true) or proportionally assigned
        Returns
        ----------
        x0 : ndarray
            Initial primary production gain, time constant and waterflood gains
            and time constants, as one long 1-d array
        """
        if tau_selection is not None:
            self.tau_selection = tau_selection

        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        d_t = self.time[1] - self.time[0]
        n_gains, n_tau, n_primary = self._opt_numbers()[:3]

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
            tau_guess1 = d_t * np.ones((n_prod, 1))

        if self.primary:
            x0 = [
                np.concatenate(
                    [
                        gains_guess1[i, :],
                        tau_guess1[i, :],
                        gains_producer_guess1[[i]],
                        tau_producer_guess1[[i]],
                    ]
                )
                for i in range(n_prod)
            ]
        else:
            x0 = [np.concatenate([gains_guess1[i, :], tau_guess1[i, :]]) for i in range(n_prod)]
        return x0

    def _opt_numbers(self) -> tuple[int, int, int]:
        """Return the number of gains, taus, and primary production parameters to fit."""
        n_gains = self.injection.shape[1]
        if self.tau_selection == "per-pair":
            n_tau = n_gains
        else:
            n_tau = 1
        if self.primary:
            n_primary = 2
        else:
            n_primary = 0
        return n_gains, n_tau, n_primary

    def _get_bounds(self, constraints: str = "") -> tuple[tuple, tuple | dict]:
        """Create bounds for the model from initialized constraints."""
        if constraints:
            self.constraints = constraints

        n_inj = self.injection.shape[1]
        n = sum(self._opt_numbers())

        if self.constraints == "positive":
            bounds = ((0, np.inf),) * n
            constraints_optimizer = ()  # type: Union[Tuple, dict]
        elif self.constraints == "sum-to-one":
            bounds = ((0, np.inf),) * n

            def constrain(x):
                x = x[:n_inj]
                return np.sum(x) - 1

            constraints_optimizer = {"type": "eq", "fun": constrain}
        elif self.constraints == "sum-to-one injector":
            raise NotImplementedError("sum-to-one injector is not implemented")
        elif self.constraints == "up-to one":
            lb = np.full(n, 0)
            ub = np.full(n, np.inf)
            ub[:n_inj] = 1
            bounds = tuple(zip(lb, ub))
            constraints_optimizer = ()
        else:
            bounds = ((0, np.inf),) * n
            constraints_optimizer = ()
        return bounds, constraints_optimizer

    def _calculate_qhat(
        self,
        x: np.ndarray,
        production: np.ndarray,
        injection: np.ndarray,
        time: np.ndarray,
    ):
        gains, tau, gain_producer, tau_producer = self._split_opts(x)
        if self.primary:
            q_hat = q_primary(production, time, gain_producer, tau_producer)
        else:
            q_hat = np.zeros(len(time))

        q_hat += self.q_CRM(injection, time, gains, tau)
        return q_hat

    def _split_opts(self, x: np.ndarray):
        n_inj = self.injection.shape[1]
        # n_prod = self.production.shape[1]
        n_gains, n_tau, n_primary = self._opt_numbers()

        gains = x[:n_inj]
        if self.tau_selection == "per-pair":
            tau = x[n_inj : n_inj * 2]
        else:
            tau = x[n_inj]
        if self.primary:
            gain_producer = x[-2]
            tau_producer = x[-1]
        else:
            gain_producer = 0
            tau_producer = 1
        if self.tau_selection == "per-pair":
            tau[tau < 1e-10] = 1e-10
        elif tau < 1e-10:
            tau = 1e-10
        if tau_producer < 1e-10:
            tau_producer = 1e-10
        return gains, tau, gain_producer, tau_producer


class CrmCompensated(CRM):
    """Bottom-hole pressure compensated CRM."""

    def fit(
        self,
        production: ndarray,
        pressure: ndarray,
        injection: ndarray,
        time: ndarray,
        initial_guess: ndarray = None,
        num_cores: int = 1,
        random: bool = False,
        **kwargs,
    ):
        """Fit a CRM model from the production, pressure, and injection data.

        Args
        ----------
        production : ndarray
            production rates for each time period,
            shape: (n_time, n_producers)
        pressure : ndarray
            average pressure for each producer for each time period,
            shape: (n_time, n_producers)
        injection : ndarray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        time : ndarray
            relative time for each rate measurement, starting from 0,
            shape: (n_time)
        initial_guess : ndarray
            initial guesses for gains, taus, primary production
            contribution
            shape: (len(guess), n_producers)
        num_cores (int): number of cores to run fitting procedure on, defaults to 1
        random : bool
            whether to randomly initialize the gains
        **kwargs:
            keyword arguments to pass to scipy.optimize fitting routine

        Returns
        ----------
        self: trained model
        """
        _validate_inputs(production, injection, time, pressure)
        self.production = production
        self.injection = injection
        self.time = time
        self.pressure = pressure

        if not initial_guess:
            initial_guess = self._get_initial_guess(random=random)
        bounds, constraints = self._get_bounds()

        def fit_well(production, pressure_local, x0):
            # residual is an L2 norm
            def residual(x, production):
                return sum(
                    (
                        production
                        - self._calculate_qhat(
                            x, production, injection, time, pressure_local, pressure
                        )
                    )
                    ** 2
                )

            result = optimize.minimize(
                residual,
                x0,
                bounds=bounds,
                constraints=constraints,
                args=(production,),
                **kwargs,
            )
            return result

        if num_cores == 1:
            results = map(fit_well, self.production.T, pressure.T, initial_guess)
        else:
            results = Parallel(n_jobs=num_cores)(
                delayed(fit_well)(prod, pressure, x0)
                for prod, pressure, x0 in zip(self.production.T, pressure.T, initial_guess)
            )

        opts_perwell = [self._split_opts(r["x"]) for r in results]
        gains_perwell, tau_perwell, gains_producer, tau_producer, gain_pressure = map(
            list, zip(*opts_perwell)
        )

        self.gains = np.vstack(gains_perwell)
        self.tau = np.vstack(tau_perwell)
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)
        self.gain_pressure = np.vstack(gain_pressure)
        return self

    def _calculate_qhat(  # TODO: start here
        self,
        x: np.ndarray,
        production: np.ndarray,
        injection: np.ndarray,
        time: np.ndarray,
        pressure_local: np.ndarray,
        pressure: np.ndarray,
    ):
        gains, tau, gain_producer, tau_producer, gain_pressure = self._split_opts(x)
        if self.primary:
            q_hat = q_primary(production, time, gain_producer, tau_producer)
        else:
            q_hat = np.zeros(len(time))

        q_hat += self.q_CRM(injection, time, gains, tau)
        q_hat += q_bhp(pressure_local, pressure, gain_pressure)
        return q_hat

    def _opt_numbers(self) -> tuple[int, int, int, int]:
        n_gain, n_tau, n_primary = super()._opt_numbers()
        return n_gain, n_tau, n_primary, self.production.shape[1]

    def _split_opts(self, x: np.ndarray) -> tuple[ndarray, ndarray, Any, Any, ndarray]:
        n_gains, n_tau, n_primary = self._opt_numbers()[:3]
        n_connectivity = n_gains + n_tau

        gains = x[:n_gains]
        tau = x[n_gains:n_connectivity]
        if self.primary:
            gain_producer = x[n_connectivity:][0]
            tau_producer = x[n_connectivity:][1]
        else:
            gain_producer = 0
            tau_producer = 1
        gain_pressure = x[n_connectivity + n_primary :]

        # boundary setting
        if self.tau_selection == "per-pair":
            tau[tau < 1e-10] = 1e-10
        elif tau < 1e-10:
            tau = 1e-10
        if tau_producer < 1e-10:
            tau_producer = 1e-10
        return gains, tau, gain_producer, tau_producer, gain_pressure

    def _get_initial_guess(self, tau_selection: str | None = None, random=False):
        """Make the initial guesses for the CRM model parameters.

        :meta private:

        Args
        ----------
        tau_selection : str, one of 'per-pair' or 'per-producer'
            sets whether to use CRM (per-pair) or CRMp model

        Returns
        ----------
        x0 : ndarray
            Initial primary production gain, time constant and waterflood gains
            and time constants, as one long 1-d array
        """
        guess = super()._get_initial_guess(tau_selection=tau_selection, random=random)
        _, _, _, n_pressure = self._opt_numbers()
        pressure_guess = np.ones(n_pressure)
        guess = [np.concatenate([guess[i], pressure_guess]) for i in range(len(guess))]
        return guess


def _validate_inputs(
    production: ndarray | None = None,
    injection: ndarray | None = None,
    time: ndarray | None = None,
    pressure: ndarray | None = None,
) -> None:
    """Validate shapes and values of inputs.

    Args
    ----
    production : ndarray, optional
    injection : ndarray, optional
    time : ndarray, optional
    pressure : ndarray, optional

    Raises
    ------
    ValueError if timesteps don't match or production and pressure don't match
    """
    inputs = {
        "production": production,
        "injection": injection,
        "time": time,
        "pressure": pressure,
    }
    inputs = {key: val for key, val in inputs.items() if val is not None}
    # Shapes
    test_prod_inj_timesteps = production is not None and injection is not None
    if test_prod_inj_timesteps and (production.shape[0] != injection.shape[0]):
        raise ValueError("production and injection do not have the same number of timesteps")
    if time is not None:
        for timeseries in inputs:
            if inputs[timeseries].shape[0] != time.shape[0]:
                raise ValueError(f"{timeseries} and time do not have the same number of timesteps")
    if production is not None:
        if (injection is not None) and (production.shape[0] != injection.shape[0]):
            raise ValueError("production and injection do not have the same number of timesteps")
        if (pressure is not None) and (production.shape != pressure.shape):
            raise ValueError("production and pressure are not of the same shape")
    if (
        (injection is not None)
        and (pressure is not None)
        and (injection.shape[0] != pressure.shape[0])
    ):
        raise ValueError("injection and pressure do not have the same number of timesteps")
    # Values
    for timeseries in inputs:
        if np.any(np.isnan(inputs[timeseries])):
            raise ValueError(f"{timeseries} cannot have NaNs")
        if np.any(inputs[timeseries] < 0.0):
            raise ValueError(f"{timeseries} cannot be negative")
