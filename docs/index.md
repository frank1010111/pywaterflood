# `pywaterflood`: Waterflood Connectivity Analysis

`pywaterflood` provides tools for capacitance resistance modeling, a
physics-inspired model for estimating waterflood performance. It estimates the
connectivities and time decays between injectors and producers. With pressure
data, it can compensate for changing bottom-hole conditions at the producers.

With no data, or in the planning stage, `pywaterflood.calc_gains_homogeneous`
can very quickly predict connectivities assuming a homogeneous, rectangular
reservoir.

## Installing

You can install this package from PyPI with the line

```
pip install pywaterflood
```

```{toctree}
:maxdepth: 1
example.ipynb
autoapi/index
changelog.md
contributing.md
conduct.md
```
