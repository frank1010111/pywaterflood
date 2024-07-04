---
title: "Pywaterflood: Well connectivity analysis through capacitance-resistance modeling"
tags:
  - Python
  - well connectivity analysis
  - waterfloods
  - CO2 floods
  - Geothermal energy
  - multiphase flow
authors:
  - name: Frank Male
    orcid: 0000-0002-3402-5578
    corresponding: true
    affiliation: 1
affiliations:
  - name: Pennsylvania State University, University Park, PA, USA
    index: 1
date: 16 September 2023
bibliography: paper.bib
---

# Summary

Well connectivity analysis has many applications for subsurface energy, covering any project where nearby wells are expected to influence one another, whether they are in oil or gas fields, geothermal fields, or an aquifer. After completing a well connectivity analysis, reservoir managers know the degree and time-dependence of influence of one well's behavior on another. With this knowledge, they can better allocate injections and plan well interventions. Capacitance Resistance Models are useful for performing well connectivity analysis with limited information about the geology of the reservoirs involved [@yousef2006capacitance]. They are called Capacitance Resistance Models to refer to the fact that the equations describing well influence mimic a network of capacitors and resistors [@holanda2018].

`Pywaterflood` is a Python package that uses Capacitance Resistance Modeling (CRM) to estimate well connectivity. It is a portmanteau of "Python" and "waterflood," where a waterflood is an oil reservoir with water injection designed to increase reservoir pressure and move oil towards producing wells. The `CRM` submodule forms the bulk of this package. It performs CRM with differing levels of complexity, where:

- producing and injecting wells share one universal time constant,
- each producer has the same time constant with each injector influencing it, or
- each producer-injector pair has a unique time constant.

CRM was developed by @yousef2006capacitance.

The `MPI` (Multiwell Productivity Index) submodule uses a geometrical model of well influence [@valko2000development], extended and applied to reservoirs with both injecting and producing wells [@kaviani2010inferring]. As a geometrical model, it can assist in planning reservoirs before any production or injection has begun.

# Statement of need

Interwell connectivity analysis is important for understanding the geology of subsurface systems. This can be used to improve oil recovery efficiency [@albertoni2003inferring], better sequester CO$_2$ [@tao2015optimizing], and optimize geothermal fields [@akin2014optimization]. @holanda2018 enumerate four uses for CRM results:

1. Finding sealing faults and high-flow-connectivity pathways
2. Investigating connectivity between adjacent reservoirs and reservoir compartments
3. Measuring the per-well effectiveness of fluid injection
4. Optimizing injection, either through redirecting fluid to different wells or to inform the placement of new wells

`Pywaterflood` uses a reduced-physics model to match connections between injecting and producing wells. As explained in @holanda2018, CRM provides a method for connectivity analysis that is more sophisticated than empirical decline analysis but also more approachable than full reservoir simulation.

There is another publicly available tool for CRM analysis of reservoirs like `pywaterflood`: @sayarpour2008development. However, that tool comes in the form of an Excel workbook with no associated license. This python package, with performance parts written in Rust, provides more extensibility and better performance than an Excel file. There are other programs for performing waterflood analysis with CRM in the industry, but they are not open sourced and publicly available. A survey of Github reveals the following examples of CRM: [a matlab script](https://github.com/billalaslam/crmwaterflood_matlab), [a proxy-CRM model "highly inspired by" pywaterflood](https://github.com/leleony/proxy-crm), and [another python script](https://github.com/deepthisen/CapacitanceResistanceModel).

The `pywaterflood` library can perform the following tasks:

1. Estimate connectivity between wells in fluid or pressure communication with `CRM`
2. History-match and forecast the production of wells in waterfloods, CO$_2$ floods, or geothermal fields with `CRM`
3. Provide purely geometric estimates of well connectivity before production data is available with `MPI`

In the period from 22 January 2024 to 21 February 2024, the pywaterflood package was downloaded from PyPI 772 times. It has been used for the author's work in analyzing waterfloods in two papers in preparation.

# Background

The governing equation for CRM to predict production ($q$) at a particular time is

$$q(t_n) = q(t_0) e^{ - \left( \frac{{t_n - t_0}}{{\tau}} \right)} + \sum_{i}\sum_{k=1}^{n} \left( \left(1 - e^{ - \frac{{\Delta \Delta t_k}}{{\tau_i}}} \right) \left( w_i(t_k) - J_i \tau_i \frac{{\Delta p_{i}(t_k)}}{{\Delta t_k}} \right) e^{ - \frac{{t_n - t_k}}{{\tau_i}}} \right).$$

It has three components that feed $q(t_n)$, the production from a well at the n'th period in time:

- $q(t_0)$: production from fluid expansion, decaying exponentially
- $w_i(t_k)$: injected fluid for the previous periods for the i'th injector
- $\Delta p_{i}$: changes in pressure for previous periods for the i'th injector

# Acknowledgements

I am thankful for Jerry Jensen and Larry Lake for their mentorship, introduction to Capacitance-Resistance Modeling, and presentation of several interesting problems for `pywaterflood`. Ian Duncan was responsible for useful discussions and further problems to apply CRM to. Danial Kaviani provided advice for the MPI submodule. Software development funding was provided by the Department of Energy grant "Optimizing Sweep based on Geochemical and Reservoir Characterization of the Residual Oil Zone of Hess Seminole Unit" (Principal investigator: Ian Duncan), the State of Texas Advanced Resource Recovery program (PI: William Ambrose, then Lorena Moscardelli), and by Pennsylvania State University faculty funds.

This project relies on the following open-source Python packages: NumPy [@numpy2011; @numpy2020], SciPy [@scipy2020], and pandas [@pandas2010]. It also uses the Rust crates ndarray, numpy, and pyo3.

# References
