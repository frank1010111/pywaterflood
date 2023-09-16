# `pywaterflood`: Waterflood Connectivity Analysis

```{toctree}
:maxdepth: 1
:hidden:
getting-started/index
user-guide/index
autoapi/index
changelog.md
contributing.md
conduct.md
```

:::::{grid} 1 1 2 2
:class-container: intro-grid text-center

::::{grid-item-card}
:columns: 12

`pywaterflood` provides tools for capacitance resistance modeling, a
physics-inspired model for estimating waterflood performance. It estimates the
connectivities and time decays between injectors and producers. With pressure
data, it can compensate for changing bottom-hole conditions at the producers.

With no data, or in the planning stage, `pywaterflood.calc_gains_homogeneous`
can very quickly predict connectivities assuming a homogeneous, rectangular
reservoir.

:::{image} https://img.shields.io/badge/frank1010111--pywaterflood-lightgrey?style=for-the-badge&logo=GitHub
:alt: GitHub
:target: https://github.com/frank1010111/pywaterflood
:class: shield-badge

:::

:::{image} https://img.shields.io/badge/-Try%20It%21-orange?style=for-the-badge
:alt: Try It!
:target: https://colab.research.google.com/github/frank1010111/pywaterflood/blob/master/docs/user-guide/7-minutes-to-pywaterflood.ipynb
:class: shield-badge
:::

::::

::::{grid-item-card}
:link-type: doc
:link: getting-started/index

{fas}`running`

Getting started
^^^^^^^^^^^^^^^

New to _pywaterflood_? Unsure what it can be used for? Check out the getting started
guides. They contain an introduction to _pywaterflood_'s features.

::::

:::{grid-item-card}
:link-type: doc
:link: user-guide/index

{fas}`book-open`

User guide
^^^^^^^^^^

The user guide provides in-depth documentation on library features for
_pywaterflood_.

:::

:::{grid-item-card}
:link-type: doc
:link: autoapi/index

{fas}`code`

API reference
^^^^^^^^^^^^^

The reference guide contains a detailed description of the functions, modules,
and objects included in _pywaterflood_. The reference describes how the methods work
and which parameters can be used.

:::

:::{grid-item-card}
:link: https://github.com/frank1010111/pywaterflood/blob/master/CONTRIBUTING.md

{fas}`terminal`

Contributor's guide
^^^^^^^^^^^^^^^^^^^

Spotted a typo in the documentation?
Want to add to the codebase? The contributing guidelines will guide you through
the process of improving _pywaterflood_.

:::

:::::
