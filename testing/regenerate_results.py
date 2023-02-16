"""Regenerate results.

This is for regenerating the results that are tested against during test_predict. Every once in
a while it makes sense to regenerate it, but be careful.

functions
--------
test_predict

use
----
run `python testing/regenerate_results.py` from the project directory
"""
from __future__ import annotations

import numpy as np
from pywaterflood import CRM

# from pywaterflood.crm import CrmCompensated, q_bhp

data_dir = "testing/data/"


def reservoir_simulation_data():
    injection = np.genfromtxt(data_dir + "injection.csv", delimiter=",")
    production = np.genfromtxt(data_dir + "production.csv", delimiter=",")
    time = np.genfromtxt(data_dir + "time.csv", delimiter=",")
    return injection, production, time


def _trained_model(*args, **kwargs):
    crm = CRM(*args, **kwargs)
    crm.injection, crm.production, crm.time = reservoir_simulation_data()
    crm.gains = np.genfromtxt(data_dir + "gains.csv", delimiter=",")
    if crm.tau_selection == "per-pair":
        crm.tau = np.genfromtxt(data_dir + "taus_per-pair.csv", delimiter=",")
    else:
        crm.tau = np.genfromtxt(data_dir + "taus.csv", delimiter=",")
    crm.gains_producer = np.genfromtxt(data_dir + "gain_producer.csv", delimiter=",")
    crm.tau_producer = np.genfromtxt(data_dir + "tau_producer.csv", delimiter=",")
    return crm


def test_predict(primary, tau_selection):
    """Predictions for test cases."""
    crm = _trained_model(primary=primary, tau_selection=tau_selection)
    return crm.predict()


if __name__ == "__main__":
    print("Regenerating prediction files")  # noqa: T201
    primary = (True, False)
    tau_selection = ("per-pair", "per-producer")
    for p in primary:
        for t in tau_selection:
            prediction = test_predict(p, t)
            primary_str = "primary" if p else "noprimary"
            np.savetxt(f"{data_dir}prediction_{primary_str}_{t}.csv", prediction, delimiter=",")
    print("Finished")  # noqa: T201
