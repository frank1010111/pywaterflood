# `pywaterflood`: Waterflood Connectivity Analysis

[![PyPI version](https://badge.fury.io/py/pywaterflood.svg)](https://badge.fury.io/py/pywaterflood)
[![Documentation Status](https://readthedocs.org/projects/pywaterflood/badge/?version=latest)](https://pywaterflood.readthedocs.io/en/latest/?badge=latest)

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![codecov](https://codecov.io/gh/frank1010111/pywaterflood/branch/master/graph/badge.svg?token=3XRGLKO7T8)](https://codecov.io/gh/frank1010111/pywaterflood)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Python version](https://img.shields.io/badge/Python-3.7%2C%203.8%2C%203.9-blue)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pywaterflood)](https://pypi.org/project/pywaterflood/)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frank1010111/pywaterflood/master?labpath=docs%2Fexample.ipynb)

`pywaterflood` provides tools for capacitance resistance modeling, a
physics-inspired model for estimating waterflood performance. It estimates the
connectivities and time decays between injectors and producers.

## Overview

A literature review has been written by Holanda, Gildin, Jensen, Lake and Kabir,
entitled "A State-of-the-Art Literature Review on Capacitance Resistance Models
for Reservoir Characterization and Performance Forecasting."
[They](https://doi.org/10.3390/en11123368) describe CRM as the following:

> The Capacitance Resistance Model (CRM) is a fast way for modeling and
> simulating gas and waterflooding recovery processes, making it a useful tool
> for improving flood management in real-time. CRM is an input-output and
> material balance-based model, and requires only injection and production
> history, which are the most readily available data gathered throughout the
> production life of a reservoir.

There are several CRM versions (see Holanda et al., 2018). Through passing
different parameters when creating the CRM instance, you can choose between
CRMIP, where a unique time constant is used for each injector-producer pair, and
CRMP, where a unique time constant is used for each producer. CRMIP is more
reliable given sufficient data. With CRMP, you can reduce the number of
unknowns, which is useful if available production data is limited.

## Getting started

You can install this package from PyPI with the line

```
pip install pywaterflood
```

### A simple example

    import pandas as pd
    from pywaterflood import CRM

    gh_url = "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
    prod = pd.read_csv(gh_url + 'production.csv', header=None).values
    inj = pd.read_csv(gh_url + "injection.csv", header=None).values
    time = pd.read_csv(gh_url + "time.csv", header=None).values[:,0]

    crm = CRM(tau_selection='per-pair', constraints='up-to one')
    crm.fit(prod, inj, time)
    q_hat = crm.predict()
    residuals = crm.residual()
