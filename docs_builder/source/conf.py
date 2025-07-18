# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# print("sys.path[0] =", sys.path[0])

project = 'Finite Automata'
copyright = '2025, Jaskirat Singh Maskeen'
author = 'Jaskirat Singh Maskeen'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_baseurl = "https://jsmaskeen.github.io/finite-automata"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",   
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_type_aliases = {
    'STATE': 'src.custom_types.STATE',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
