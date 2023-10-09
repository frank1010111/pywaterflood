# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

from datetime import datetime

# -- Project information -----------------------------------------------------
project = "pywaterflood"
copyright = f"2021-{datetime.now().year}, Frank Male"
author = "Frank Male"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
autoapi_dirs = ["../python/pywaterflood"]
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
    "colon_fence",
    "deflist",
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
html_theme = "furo"
html_static_path: list[str] = []
html_css_files = ["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"]
