import os.path as osp
from pathlib import Path

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(Path.home(), "benchmark", "data", "experiments")

MODELS_DIR = osp.join(Path.home(), "benchmark", "data", "models")

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False
