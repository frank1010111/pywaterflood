import numpy as np
from numpy import ndarray
from numba import njit
import pandas as pd
import pickle
from scipy import optimize
from typing import Optional, Tuple, Union
from joblib import Parallel, delayed


@njit
def q_primary(
    production: ndarray, time: ndarray, gain_producer: ndarray, tau_producer: ndarray
) -> ndarray:
    """Calculates primary production contribution using Arps equation with b=0

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
def q_CRM_perpair(
    injection: ndarray, time: ndarray, gains: ndarray, taus: ndarray
) -> ndarray:
    """Calculates per injector-producer pair production for all injectors on one producer
    using CRM model

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
        conv_injected[0, j] += (1 - np.exp((time[0] - time[1]) / taus[j])) * injection[
            0, j
        ]
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
def q_CRM_perproducer(
    injection: ndarray, time: ndarray, gain: ndarray, tau: float
) -> ndarray:
    """Calculates per injector-producer pair production for all injectors on one producer
    using simplified CRMp model that assumes a single tau for each producer

    Args
    ----------
    injection : ndarray
        Production, size: Number of time steps
    time : ndarray
        Producing times to forecast, size: Number of time steps
    gains : ndarray
        Connectivities between each injector and the producer, 
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


def random_weights(
    n_i: int, n_j: int, axis: int = 0, seed: Optional[int] = None
) -> ndarray:
    """Generates random weights for producer-injector gains
    
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
    """A Capacitance Resistance Model history matcher

    CRM uses a physics-inspired mass balance approach to explain production for \
        waterfloods. It treants each injector-producer well pair as a system \
        with mass input, output, and pressure related to the mass balance. \
        Several versions exist.

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
        """Build a CRM model from the production and injection data (production, injection)

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
            initial guesses for gains, taus, primary production
            contribution, of 
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

        if not initial_guess:
            initial_guess = self._get_initial_guess(random=random)
        bounds, constraints = self._get_bounds()
        num_cores = kwargs.pop("num_cores", 1)

        def fit_well(production, x0):
            # residual is an L2 norm
            def residual(x, production):
                return sum(
                    (production - self._calculate_qhat(x, production, injection, time))
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

        production_perwell = [x for x in self.production.T]
        if num_cores == 1:
            results = map(fit_well, production_perwell, initial_guess)
        else:
            results = Parallel(n_jobs=num_cores)(
                delayed(fit_well)(p) for p, x0 in zip(production_perwell, initial_guess)
            )

        opts_perwell = [self._split_opts(r["x"]) for r in results]
        gains_perwell, tau_perwell, gains_producer, tau_producer = map(
            list, zip(*opts_perwell)
        )

        self.gains = np.vstack(gains_perwell)
        self.tau = np.vstack(tau_perwell)
        self.gains_producer = np.array(gains_producer)
        self.tau_producer = np.array(tau_producer)
        return self

    def predict(self, injection=None, time=None, connections={}):
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
            q_hat[:, i] += q_primary(
                production[:, i], time, gains_producer[i], tau_producer[i]
            )
            q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
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
        """Write trained model to an Excel file
        
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
        """Write trained model to a pickle file
        
        Args
        -----
        fname : str
            pickle file to write out
        """
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def _get_initial_guess(self, tau_selection: str = "", random=False):
        """Creates the initial guesses for the CRM model parameters

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
            x0 = [
                np.concatenate([gains_guess1[i, :], tau_guess1[i, :]])
                for i in range(n_prod)
            ]
        return x0

    def _opt_numbers(self) -> Tuple[int, int, int]:
        """
        returns the number of gains, taus, and primary production parameters to fit
        """
        n_inj = self.injection.shape[1]
        # n_prod = self.production.shape[1]
        n_gains = n_inj
        if self.tau_selection == "per-pair":
            n_tau = n_gains
        else:
            n_tau = 1
        if self.primary:
            n_primary = 2
        else:
            n_primary = 0
        return n_gains, n_tau, n_primary

    def _get_bounds(self, constraints: str = "") -> Tuple[Tuple, Union[Tuple, dict]]:
        """Create bounds for the model from initialized constraints
        """
        if constraints:
            self.constraints = constraints

        n_inj = self.injection.shape[1]
        if self.tau_selection == "per-pair":
            n = n_inj * 2
        else:
            n = n_inj + 1

        if self.primary:
            n = n + 2

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
        ":meta private:"
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
