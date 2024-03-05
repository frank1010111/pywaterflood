# `pywaterflood`: Waterflood Connectivity Analysis

[![PyPI version](https://badge.fury.io/py/pywaterflood.svg)](https://badge.fury.io/py/pywaterflood)
[![Conda](https://img.shields.io/conda/v/conda-forge/pywaterflood)](https://anaconda.org/conda-forge/pywaterflood)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pywaterflood)](https://pypi.org/project/pywaterflood/)

[![Documentation Status](https://readthedocs.org/projects/pywaterflood/badge/?version=latest)](https://pywaterflood.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/234408267.svg)](https://zenodo.org/badge/latestdoi/234408267)
[![status](https://joss.theoj.org/papers/2fdffa96e936553d289e622e5e12388c/status.svg)](https://joss.theoj.org/papers/2fdffa96e936553d289e622e5e12388c)

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![codecov](https://codecov.io/gh/frank1010111/pywaterflood/branch/master/graph/badge.svg?token=3XRGLKO7T8)](https://codecov.io/gh/frank1010111/pywaterflood)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

`pywaterflood` provides tools for capacitance resistance modeling, a
physics-inspired model for estimating well connectivity between injectors and
producers or producers and other producers. It is useful for analyzing and
optimizing waterfloods, CO<sub>2</sub> floods, and geothermal projects.

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

Or from conda/mamba with

```
conda install -c conda-forge pywaterflood
```

Then, [read the docs](https://pywaterflood.readthedocs.io/) to learn more. If you
want to try it out online before installing it on your computer, you can run
[this google colab notebook](https://colab.research.google.com/github/frank1010111/pywaterflood/blob/master/docs/user-guide/7-minutes-to-pywaterflood.ipynb).

### A simple example

    import numpy as np
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

    print("MAE by well:", np.round(np.abs(residuals).mean(axis=0), 2), "barrels")
    print("MAPE by well:", np.round(np.mean(np.abs(residuals) / prod * 100, axis=0), 2), "percent")
    print("RMSE by well:", np.round(np.sqrt(np.sum(residuals**2, axis=0)), 2))

## Contributing

Contributions are extremely welcome! Have [an issue to report](https://github.com/frank1010111/bluebonnet/issues/new)?
Want to offer new features or documentation? Check out the [contribution guide](https://github.com/frank1010111/pywaterflood/blob/master/CONTRIBUTING.md)
to help you set up. Discussions could start anytime at
[the discussions section](https://github.com/frank1010111/pywaterflood/discussions).

`pywaterflood` uses Rust for computation and python as the high level interface.
Luckily, [maturin](https://www.maturin.rs/) is a very convenient tool for working
with mixed Python-Rust projects.

Running tests, building the package, linting to conform to code standards, and building the documentation are all handled by [nox](https://nox.thea.codes).

### Running tests

The [guide for getting started](https://github.com/frank1010111/pywaterflood/blob/master/CONTRIBUTING.md#get-started), has instructions for installing rust, python, and nox. At that point, both the lint and unit test sessions are run with the command

```
nox
```

## License

This software library is released under a BSD 2-Clause License.

## Acknowledgments

Capacitance resistance modeling would not have caught on without the persistence
of two professors: Larry Lake and Jerry Jensen. Both of these gentlemen generously
helped answer questions in the development of this library. Research funding for
this project came from the Department of Energy grant "Optimizing Sweep based on
Geochemical and Reservoir Characterization of the Residual Oil Zone of Hess Seminole
Unit" (PI: Ian Duncan) and the State of Texas Advanced Resource Recovery program
(PI: William Ambrose). Further development is supported by Penn State faculty
promotion funds and volunteer time.
