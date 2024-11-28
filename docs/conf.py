"""
Fichier de configuration pour le générateur de documentation Sphinx.

Ce fichier contient uniquement une sélection des options les plus courantes.
Pour une liste complète, consultez la documentation :
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Webapp streamlit d'analyse de données"
copyright = "2024, Laury, Hugo, Jael"
author = "Laury, Hugo, Jael"
release = "0.1"

# -- Configuration générale ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "fr"

# -- Options pour l'output HTML -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
