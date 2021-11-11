# `pywaterflood`: Waterflood Connectivity Analysis
[![Documentation Status](https://readthedocs.org/projects/pywaterflood/badge/?version=latest)](https://pywaterflood.readthedocs.io/en/latest/?badge=latest)  

`pywaterflood` provides tools for capacitance resistance modeling, a physics-inspired model for estimating waterflood performance. It estimates the connectivities and time decays between injectors and producers.

## Overview

A literature review has been written by Holanda, Gildin, Jensen, Lake and Kabir, entitled "A State-of-the-Art Literature Review on Capacitance Resistance Models for Reservoir Characterization and Performance Forecasting." They describe CRM as the following:
> The Capacitance Resistance Model (CRM) is a fast way for modeling and simulating gas and waterflooding recovery processes, making it a useful tool for improving flood management in real-time. CRM is an input-output and material balance-based model, and requires only injection and production history, which are the most readily available data gathered throughout the production life of a reservoir.

## Getting started
You can install this package from PyPI with the line
```
pip install pywaterflood
```

## A simple example
    import pandas as pd
    from pywaterflood import CRM

    gh_url = https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
    prod = pd.read_csv(gh_url + 'production.csv', header=None).values
    inj = pd.read_csv(gh_url + \"injection.csv\", header=None).values
    time = pd.read_csv(gh_url + \"time.csv\", header=None).values[:,0]

    crm = CRM(tau_selection='per-pair', constraints='up-to one')
    crm.fit(prod, inj, time)
    q_hat = crm.predict()
    residuals = crm.residual()
