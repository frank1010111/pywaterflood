# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

# -- Project information -----------------------------------------------------
project = "pywaterflood"
copyright = "2021, Frank Male"
author = "Frank Male"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../pywaterflood"]
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]


# myst_nb configuration
nb_execution_mode = "off"
nb_execution_timeout = 60
nb_output_stderr = "remove"
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
