#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains general configuration settings.

The default values can be overwritten by setting the corresponding environment
variables.
"""


import os
from pathlib import Path

from offline_mbrl.utils.str2bool import str2bool

DATA_DIR = os.environ.get("OMBRL_DATA_DIR", Path.cwd() / "data" / "experiments")
"""Where experiment outputs are saved by default."""

MODELS_DIR = os.environ.get("OMBRL_MODELS_DIR", Path.cwd() / "data" / "models")
"""Where models are saved by default."""

FORCE_DATESTAMP = str2bool(os.environ.get("OMBRL_FORCE_DATESTAMP", "False"))
"""Whether to automatically insert a date and time stamp into the names of
save directories."""
