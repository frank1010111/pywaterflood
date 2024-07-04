"""Analyze waterfloods with capacitance-resistance models.

This is the central module in ``pywaterflood``, based around the :code:`CRM`
class, which implements the standard capacitance-resistance models. For most
cases, the best performance comes from selecting
:code:`CRM(primary=True, tau_selection="per-pair", constraints="up-to one")`.
In the literature, this is referred to as CRM-IP (injector producer).

If the data is too sparse, then change ``tau_selection`` to "per-producer".
This reduces the number of variables to fit by nearly half by using only one
time constant for all well connections influencing a producer. This is referred
to as CRM-P in the literature.

If the data is still too sparse, you can sum all the injectors, all the producers,
or both. This greatly decreases the utility of the model and is not recommended. In
the literature, it is known as CRM-T.

The base class assumes constant bottomhole pressures for the producing wells.
If you know the pressures for these wells or at least the trend, consider using
``CrmCompensated``.

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy import optimize

from pywaterflood import _core


def q_primary(
    production: NDArray, time: NDArray, gain_producer: float, tau_producer: float
) -> NDArray:
    r"""Calculate primary production contribution.

    Uses Arps equation with :math:`b=0`

    .. math::
        q_{p}(t) = q_i e^{-bt}

    Args
    ----------
    production : NDArray
        Production, size: Number of time steps
    time : NDArray
        Producing times to forecast, size: Number of time steps
    gain_producer : float
        Arps :math:`q_i` factor
    tau_producer : float
        Arps time constant

    Returns
    ----------
    q_hat : NDArray
        Calculated production, :math:`\hat q`, size: Number of time steps
    """
    return _core.q_primary(production, time, gain_producer, tau_producer)


def q_CRM_perpair(injection: NDArray, time: NDArray, gains: NDArray, taus: NDArray) -> NDArray:
    """Calculate per injector-producer pair production.

    Runs for influences of each injector on one producer, assuming
    individual :code:`gain` and :code:`tau` for each pair

    Args
    ----------
    injection : NDArray
        Injected fluid, size: Number of time steps
    time : NDArray
        Producing times to forecast, size: Number of time steps
    gains : NDArray
        Connectivities between each injector and the producer,
        size: Number of injectors
    taus : NDArray
        Time constants between each injector and the producer,
        size: Number of injectors

    Returns
    ----------
    q_hat : NDArray
        Calculated production :math:`\\hat q`, size: Number of time steps
    """
    return _core.q_crm_perpair(injection, time, gains, taus)


def q_CRM_perproducer(injection: NDArray, time: NDArray, gain: NDArray, tau: float) -> NDArray:
    """Calculate per injector-producer pair production (simplified tank).

    Uses simplified CRMP model that assumes a single tau for each producer

    Args
    ----------
    injection : NDArray
        injected fluid in reservoir volumes, size: Number of time steps
    time : NDArray
        Producing times to forecast, size: Number of time steps
    gains : NDArray
        Connectivities between each injector and the producer
        size: Number of injectors
    tau : float
        Time constants all injectors and the producer

    Returns
    ----------
    q_hat : NDArray
        Calculated production :math:`\\hat q`

        shape: Number of time steps
    """
    tau2 = np.full(injection.shape[1], tau)
    return q_CRM_perpair(injection, time, gain, tau2)


def q_bhp(pressure_local: NDArray, pressure: NDArray, v_matrix: NDArray) -> NDArray:
    r"""Calculate the production effect from bottom-hole pressure variation.

    This looks like

    .. math::
        q_{BHP,j}(t_i) = \sum_{k} v_{kj}\left[ p_j(t_{i-1}) - p_k(t_i) \right]

    Args
    ----
    pressure_local : NDArray
        pressure for the well in question, shape: n_time
    pressure : NDArray
        bottomhole pressure, shape: n_time, n_producers
    v_matrix : NDArray
        connectivity between one producer and all producers, shape: n_producers

    Returns
    -------
    q : NDArray
        production from changing BHP, shape: n_time
    """
    return _core.q_bhp(pressure_local, pressure, v_matrix)


def random_weights(n_prod: int, n_inj: int, axis: int = 0, seed: int | None = None) -> NDArray:
    """Generate random weights for producer-injector gains.

    Args
    ----
    n_prod : int
        Number of producing wells
    n_inj : int
        Number of injecting wells
    axis : int, default is 0
        0 corresponds to normalizing among producers, 1 to normalizing among injectors
    seed : int, default is None
        state for random number generator

    Returns
    -------
    gains_guess: NDArray
        shape: n_prod, n_inj
    """
    rng = np.random.default_rng(seed)
    limit = 10 * (n_prod if axis == 0 else n_inj)
    vec = rng.integers(0, limit, (n_prod, n_inj))
    axis_sum = vec.sum(axis, keepdims=True)
    return vec / axis_sum


class CRM:
    """A Capacitance Resistance Model history matcher.

    CRM uses a physics-inspired mass balance approach to explain production for
    waterfloods. It treats each injector-producer well pair as a system
    with mass input, output, and pressure related to the mass balance.
    Several versions exist and can be selected from the arguments.

    The default arguments give the best results for most scenarios, but they
    can be sub-optimal if there is insufficient data, and they run slower than
    models with more simplifying assumptions.

    Args
    ----------
    primary : bool
        Whether to model primary production (True is strongly recommended)
    tau_selection : str
        How many tau values to select
            - If 'per-pair', fit tau for each producer-injector pair
            - If 'per-producer', fit tau for each producer (CRMP model)
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
    >>> crm = CRM(True, "per-pair", "up-to one")

    References
    ----------
    "A State-of-the-Art Literature Review on Capacitance Resistance Models for
    Reservoir Characterization and Performance Forecasting" - Wanderley de Holanda
    et al., 2018. https://www.mdpi.com/1996-1073/11/12/3368
    """

    def __init__(
        self,
        primary: bool = True,
        tau_selection: str = "per-pair",
        constraints: str = "positive",
    ):
        """Initialize CRM with appropriate settings."""
        if not isinstance(primary, bool):
            msg = "primary must be a boolean"
            raise TypeError(msg)
        self.primary = primary
        if constraints not in (
            "positive",
            "up-to one",
            "sum-to-one",
            "sum-to-one injector",
        ):
            msg = "Invalid constraints"
            raise ValueError(msg)
        self.constraints = constraints
        self.tau_selection = tau_selection
        if tau_selection == "per-pair":
            self.q_CRM = q_CRM_perpair
        elif tau_selection == "per-producer":
            self.q_CRM = q_CRM_perproducer
        else:
            msg = f'tau_selection must be one of ("per-pair","per-producer"), not {tau_selection}'
            raise ValueError(msg)

    def fit(
        self,
        production: NDArray,
        injection: NDArray,
        time: NDArray,
        initial_guess: NDArray = None,
        num_cores: int = 1,
        random: bool = False,
        **kwargs,
    ):
        """Build a CRM model from the production and injection data.

        Args
        ----------
        production : NDArray
            production rates for each time period,
            shape: (n_time, n_producers)
        injection : NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        time : NDArray
            relative time for each rate measurement, starting from 0,
            shape: (n_time)
        initial_guess : NDArray
            initial guesses for gains, taus, primary production contribution
            shape: (len(guess), n_producers)
        num_cores : int
            number of cores to run fitting procedure on, defaults to 1
        random : bool
            whether to randomly initialize the gains
        **kwargs:
            keyword arguments to pass to scipy.optimize fitting routine

        Returns
        ----------
        self: trained model

        Example
        -------
        >>> gh_url = (
        ...     "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
        ... )
        >>> prod = pd.read_csv(gh_url + "production.csv", header=None).values
        >>> inj = pd.read_csv(gh_url + "injection.csv", header=None).values
        >>> time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
        >>> crm = CRM(True, "per-pair", "up-to one")
        >>> crm.fit(prod, inj, time)
        """
        _validate_inputs(production, injection, time)
        self.production = production
        self.injection = injection
        self.time = time

        if not initial_guess:
            initial_guess = self._get_initial_guess(random=random)
        bounds, constraints = self._get_bounds()

        def fit_well(production, x0):
            # residual is an L2 norm
            def residual(x, production):
                return sum(
                    (production - self._calculate_qhat(x, production, injection, time)) ** 2
                )

            return optimize.minimize(
                residual,
                x0,
                bounds=bounds,
                constraints=constraints,
                args=(production,),
                **kwargs,
            )

        if num_cores == 1:
            results = map(fit_well, self.production.T, initial_guess)
        else:
            results = Parallel(n_jobs=num_cores)(
                delayed(fit_well)(p, x0) for p, x0 in zip(self.production.T, initial_guess)
            )

        opts_perwell = [self._split_opts(r["x"]) for r in results]
        gains_perwell, tau_perwell, gains_producer, tau_producer = map(list, zip(*opts_perwell))

        self.gains: NDArray = np.vstack(gains_perwell)
        self.tau: NDArray = np.vstack(tau_perwell)
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)
        return self

    def predict(self, injection=None, time=None, connections=None, production=None):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection : Optional NDArray
            The injection rates to input to the system, shape (n_time, n_inj)
        time : Optional NDArray
            The timesteps to predict
        connections : Optional dict
            if present, the gains, tau, gains_producer, tau_producer matrices
        production : Optional NDArray
            The production (only takes first row to use for primary production decline)

        Returns
        ----------
        q_hat :NDArray
            The predicted values, shape (n_time, n_producers)

        Example
        -------
        Using the synthetic reservoir:

        >>> gh_url = (
        ...     "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
        ... )
        >>> prod = pd.read_csv(gh_url + "production.csv", header=None).values
        >>> inj = pd.read_csv(gh_url + "injection.csv", header=None).values
        >>> time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
        >>> crm = CRM(True, "per-producer", "up-to one")
        >>> crm.fit(prod, inj, time)
        >>> crm.predict()

        Starting from a known model:

        >>> injection = np.ones((100, 2))
        >>> production = np.ones((1, 1)) * 2
        >>> time = np.arange(100, dtype=float)
        >>> connections = {
        ...     "gains": np.ones((2, 1)) * 0.95,
        ...     "tau": np.ones((2, 1)) * 3,
        ...     "gains_producer": np.zeros(1),
        ...     "tau_producer": np.ones(1),
        ... }
        >>> crm = CRM(False, "per-pair")
        >>> crm.predict(injection, time, connections=connections, production=production)
        """
        if production is None:
            production = self.production
        n_producers = production.shape[1]

        if connections is not None:
            gains = connections.get("gains")
            if gains is None:
                gains = self.gains
            tau = connections.get("tau")
            if tau is None:
                tau = self.tau
            gains_producer = connections.get("gains_producer")
            if gains_producer is None:
                gains_producer = self.gains_producer if self.primary else np.zeros(n_producers)
            tau_producer = connections.get("tau_producer")
            if tau_producer is None:
                tau_producer = self.tau_producer if self.primary else np.ones(n_producers)
        else:
            gains = self.gains
            tau = self.tau
            gains_producer = self.gains_producer
            tau_producer = self.tau_producer

        if int(injection is None) + int(time is None) == 1:
            msg = "Either both or neither of injection or time must be specified"
            raise TypeError(msg)
        if injection is None:
            injection = self.injection
        if time is None:
            time = self.time
        if time.shape[0] != injection.shape[0]:
            msg = "injection and time need same number of steps"
            raise ValueError(msg)

        q_hat = np.zeros((len(time), n_producers))
        for i in range(n_producers):
            if self.primary:
                q_hat[:, i] += q_primary(
                    production[:, i], time, gains_producer[i], tau_producer[i]
                )
            q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
        return q_hat

    def set_rates(self, production=None, injection=None, time=None):
        """Set production and injection rates and time array.

        Args
        -----
        production : NDArray
            production rates with shape (n_time, n_producers)
        injection : NDArray
            injection rates with shape (n_time, n_injectors)
        time : NDArray
            timesteps with shape n_time

        Example
        -------
        >>> injection = np.ones((100,2))
        >>> production = np.full((100,1), 2.0)
        >>> time = np.arange(100, dtype=float)
        >>> crm = CRM()
        >>> crm.set_rates(production, injection, time)
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
        gains : NDArray
            connectivity between injector and producer
            shape: n_gains, n_producers
        tau : NDArray
            time-constant for injection to be felt by production
            shape: either n_producers or (n_gains, n_producers)
        gains_producer : NDArray
            gain on primary production, shape: n_producers
        tau_producer : NDArray
            Arps time constant for primary production, shape: n_producers

        Example
        -------
        >>> crm = CRM(False, "per-pair")
        >>> gains = np.full((2, 1),0.95)
        >>> tau = np.full((2, 1), 3.0)
        >>> crm.set_connections(gains, tau)
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
        production : NDArray
            The production rates observed, shape: (n_timesteps, n_producers)
        injection : NDArray
            The injection rates to input to the system,
            shape: (n_timesteps, n_injectors)
        time : NDArray
            The timesteps to predict

        Returns
        ----------
        residual : NDArray
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
            if x not in self.__dict__:
                msg = "Model has not been trained"
                raise ValueError(msg)
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
        with Path(fname).open("wb") as f:
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
        x0 : NDArray
            Initial primary production gain, time constant and waterflood gains
            and time constants, as one long 1-d array
        """
        if tau_selection is not None:
            self.tau_selection = tau_selection

        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        d_t = self.time[1] - self.time[0]

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
        n_tau = n_gains if self.tau_selection == "per-pair" else 1
        n_primary = 2 if self.primary else 0
        return n_gains, n_tau, n_primary

    def _get_bounds(self, constraints: str = "") -> tuple[tuple, tuple | dict]:
        """Create bounds for the model from initialized constraints."""
        if constraints:
            self.constraints = constraints

        n_inj = self.injection.shape[1]
        n = sum(self._opt_numbers())

        if self.constraints == "positive":
            bounds = ((0, np.inf),) * n
            constraints_optimizer = ()  # type: tuple | dict
        elif self.constraints == "sum-to-one":
            bounds = ((0, np.inf),) * n

            def constrain(x):
                x = x[:n_inj]
                return np.sum(x) - 1

            constraints_optimizer = {"type": "eq", "fun": constrain}
        elif self.constraints == "up-to one":
            lb = np.full(n, 0)
            ub = np.full(n, np.inf)
            ub[:n_inj] = 1
            bounds = tuple(zip(lb, ub))
            constraints_optimizer = ()
        elif self.constraints == "sum-to-one injector":
            msg = "sum-to-one injector is not implemented"
            raise NotImplementedError(msg)
        else:
            msg = (
                f"Constraint must be valid, not {self.constraints}.\n"
                "For least constrained, use 'positive'"
            )
            raise ValueError(msg)
        return bounds, constraints_optimizer

    def _calculate_qhat(
        self,
        x: NDArray,
        production: NDArray,
        injection: NDArray,
        time: NDArray,
    ):
        gains, tau, gain_producer, tau_producer = self._split_opts(x)
        q_hat = np.zeros(len(time))
        if self.primary:
            q_hat += q_primary(production, time, gain_producer, tau_producer)

        q_hat += self.q_CRM(injection, time, gains, tau)
        return q_hat

    def _split_opts(self, x: NDArray):
        n_inj = self.injection.shape[1]

        gains = x[:n_inj]
        tau = x[n_inj : n_inj * 2] if self.tau_selection == "per-pair" else x[n_inj]
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
        production: NDArray,
        pressure: NDArray,
        injection: NDArray,
        time: NDArray,
        initial_guess: NDArray = None,
        num_cores: int = 1,
        random: bool = False,
        **kwargs,
    ):
        """Fit a CRM model from the production, pressure, and injection data.

        Args
        ----------
        production : NDArray
            production rates for each time period,
            shape: (n_time, n_producers)
        pressure : NDArray
            average pressure for each producer for each time period,
            shape: (n_time, n_producers)
        injection : NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        time : NDArray
            relative time for each rate measurement, starting from 0,
            shape: (n_time)
        initial_guess : NDArray
            initial guesses for gains, taus, primary production
            contribution
            shape: (len(guess), n_producers)
        num_cores : int
            number of cores to run fitting procedure on, defaults to 1
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

        if initial_guess is None:
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

            return optimize.minimize(
                residual,
                x0,
                bounds=bounds,
                constraints=constraints,
                args=(production,),
                **kwargs,
            )

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

        self.gains: NDArray = np.vstack(gains_perwell)
        self.tau: NDArray = np.vstack(tau_perwell)
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)
        self.gain_pressure: NDArray = np.vstack(gain_pressure)
        return self

    def predict(
        self,
        injection=None,
        time=None,
        connections=None,
        production=None,
        pressure=None,
    ):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection : Optional NDArray
            The injection rates to input to the system, shape (n_time, n_inj)
        time : Optional NDArray
            The timesteps to predict
        connections : Optional dict
            if present, the gains, tau, gains_producer, tau_producer matrices
        production : Optional NDArray
            The production (only takes first row to use for primary production decline)

        Returns
        ----------
        q_hat :NDArray
            The predicted values, shape (n_time, n_producers)

        Example
        -------
        Using the synthetic reservoir:

        >>> gh_url = (
        ...     "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
        ... )
        >>> prod = pd.read_csv(gh_url + "production.csv", header=None).values
        >>> inj = pd.read_csv(gh_url + "injection.csv", header=None).values
        >>> time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
        >>> pressure = 1000 - prod * 0.1
        >>> crm = CrmCompensated(True, "per-producer", "up-to one")
        >>> crm.fit(prod, pressure, inj, time)
        >>> crm.predict()

        Starting from a known model:

        >>> injection = np.ones((100, 2))
        >>> production = np.ones((1, 1)) * 2
        >>> pressure = 1000 - production * 0.1
        >>> time = np.arange(100, dtype=float)
        >>> connections = {
        ...     "gains": np.ones((2, 1)) * 0.95,
        ...     "tau": np.ones((2, 1)) * 3,
        ...     "gains_producer": np.zeros(1),
        ...     "tau_producer": np.ones(1),
        ... }
        >>> crm = CRM(False, "per-pair")
        >>> crm.predict(injection, time, connections=connections, production=production)
        """
        if int(injection is None) + int(time is None) == 1:
            msg = "Either both or neither of injection or time must be specified"
            raise TypeError(msg)

        injection = self.injection if injection is None else injection
        time = self.time if time is None else time
        production = self.production if production is None else production
        pressure = self.pressure if pressure is None else pressure
        n_producers = production.shape[1]
        if time.shape[0] != injection.shape[0]:
            msg = "injection and time need same number of steps"
            raise ValueError(msg)

        if connections is not None:
            gains = connections.get("gains")
            if gains is None:
                gains = self.gains
            tau = connections.get("tau")
            if tau is None:
                tau = self.tau
            gains_producer = connections.get("gains_producer")
            if gains_producer is None:
                gains_producer = self.gains_producer if self.primary else np.zeros(n_producers)
            tau_producer = connections.get("tau_producer")
            if tau_producer is None:
                tau_producer = self.tau_producer if self.primary else np.ones(n_producers)
            gains_pressure = connections.get("gains_pressure")
            if gains_pressure is None:
                gains_pressure = self.gain_pressure
        else:
            gains = self.gains
            tau = self.tau
            gains_producer = self.gains_producer
            tau_producer = self.tau_producer
            gains_pressure = self.gain_pressure

        q_hat = np.zeros((len(time), n_producers))
        for i in range(n_producers):
            if self.primary:
                q_hat[:, i] += q_primary(
                    production[:, i], time, gains_producer[i], tau_producer[i]
                )
                q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
                q_hat[:, i] += q_bhp(pressure[:, i], pressure, gains_pressure[i, :])
        return q_hat

    def _calculate_qhat(
        self,
        x: NDArray,
        production: NDArray,
        injection: NDArray,
        time: NDArray,
        pressure_local: NDArray,
        pressure: NDArray,
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

    def _split_opts(self, x: NDArray) -> tuple[NDArray, NDArray, Any, Any, NDArray]:
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
        x0 : NDArray
            Initial primary production gain, time constant and waterflood gains
            and time constants, as one long 1-d array
        """
        guess = super()._get_initial_guess(tau_selection=tau_selection, random=random)
        _, _, _, n_pressure = self._opt_numbers()
        pressure_guess = np.ones(n_pressure)
        return [np.concatenate([guess[i], pressure_guess]) for i in range(len(guess))]


def _validate_inputs(
    production: NDArray | None = None,
    injection: NDArray | None = None,
    time: NDArray | None = None,
    pressure: NDArray | None = None,
) -> None:
    """Validate shapes and values of inputs.

    Args
    ----
    production : NDArray, optional
    injection : NDArray, optional
    time : NDArray, optional
    pressure : NDArray, optional

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
    if time is not None:
        for timeseries in inputs:
            if inputs[timeseries].shape[0] != time.shape[0]:
                msg = f"{timeseries} and time do not have the same number of timesteps"
                raise ValueError(msg)
    if production is not None:
        if (injection is not None) and (production.shape[0] != injection.shape[0]):
            msg = "production and injection do not have the same number of timesteps"
            raise ValueError(msg)
        if (pressure is not None) and (production.shape != pressure.shape):
            msg = "production and pressure are not of the same shape"
            raise ValueError(msg)
    if (
        (injection is not None)
        and (pressure is not None)
        and (injection.shape[0] != pressure.shape[0])
    ):
        msg = "injection and pressure do not have the same number of timesteps"
        raise ValueError(msg)
    # Values
    for timeseries in inputs:
        if np.any(np.isnan(inputs[timeseries])):
            msg = f"{timeseries} cannot have NaNs"
            raise ValueError(msg)
        if np.any(inputs[timeseries] < 0.0):
            msg = f"{timeseries} cannot be negative"
            raise ValueError(msg)
