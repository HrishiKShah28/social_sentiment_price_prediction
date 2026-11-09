"""HengSeng Sentiment Price Prediction Agent package."""

import os
import sys
import site

# Automatically link local .venv site-packages if present
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_site = os.path.join(_base_dir, ".venv", "Lib", "site-packages")
if os.path.exists(_venv_site):
    site.addsitedir(_venv_site)

__version__ = "0.1.0"
