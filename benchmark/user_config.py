from pathlib import Path
import os.path as osp

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(Path.home(), 'Projects', 'thesis-code', 'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

MODELS_DIR = osp.join(Path.home(), 'Projects', 'thesis-code', 'benchmark', 'models')
