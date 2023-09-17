---
title: "Pywaterflood: Well connectivity analysis through capacitance-resistance modeling"
tags:
  - Python
  - well connectivity analysis
  - waterfloods
  - CO2 floods
  - Geothermal
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

Well connectivity analysis has many applications for subsurface energy, from waterfloods to CO$_2$ floods to geothermal. Capacitance Resistance Models are useful for performing well connectivity analysis with limited information about the geology of the reservoirs involved. They are so-called because the equations describing well influence mimic a network of capacitors and resistors.

`Pywaterflood` is a Python package that uses Capacitance Resistance Modeling to estimate well connectivity. The `CRM` submodule forms the bulk of this package. It can perform capacitance resistance modeling with differing levels of complexity, from assuming that producing and injecting wells share one universal time constant, to each producer has the same time constant with all injectors, to each producer-injector pair has an its own time constant. CRM was developed by @yousef2006capacitance. The `MPI` submodule uses a geometrical model of well influence [@valko2000development], extended and applied to reservoirs with both injecting and producing wells [@kaviani2010inferring].

# Statement of need

Interwell connectivity analysis is important for understanding the geology of subsurface systems. This can be used to improve oil recovery efficiency [@albertoni2003inferring], better sequester CO$_2$ [@tao2015optimizing], and optimize geothermal fields [@akin2014optimization].

`Pywaterflood` uses a reduced-physics model to match connections between injecting and producing wells. As explained in @holanda2018, capacitance-resistance modeling provides a method for connectivity analysis more sophisticated than empirical decline analysis, but more approachable than full reservoir simulation.

There is another publicly available tool for capacitance resistance modeling reservoirs like `pywaterflood` [@sayarpour2008development]. However, that tool comes in the form of an Excel workbook and no associated license. This python package provides more extensibility and better performance than an Excel file. There are other programs for performing waterflood analysis with CRM in the industry, but they are not open sourced and available for researchers to use.

The `pywaterflood` library can perform the following tasks:

1. Estimate connectivity between wells in fluid or pressure communication
2. History-match and forecast the production of wells in waterfloods, CO$_2$ floods, or geothermal fields
3. Provide purely geometric estimates of well connectivity before production data is available

# Acknowledgements

I am thankful for Jerry Jensen and Larry Lake for their mentorship, introduction to Capacitance-Resistance modeling, and presentation of several interesting problems for CRM. Ian Duncan was responsible for useful discussions and further problems to apply CRM to. Danial Kaviani provided advice for the MPI submodule. Software development funding was provided by the Department of Energy grant "Optimizing Sweep based on Geochemical and Reservoir Characterization of the Residual Oil Zone of Hess Seminole Unit" (Principal investigator: Ian Ducan), the State of Texas Advanced Resource Recovery program (PI: William Ambrose, then Lorena Moscardelli), and by Pennsylvania State University faculty funds.

This project relies on the following open-source Python packages: NumPy [@numpy2011; @numpy2020], SciPy [@scipy2020], and pandas [@pandas2010]. It also uses the Rust crates ndarray, numpy, and pyo3.

# References
