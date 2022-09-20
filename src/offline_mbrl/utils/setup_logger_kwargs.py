# Based on https://spinningup.openai.com

import os.path as osp
import time
from pathlib import Path
from typing import Optional, Union

from offline_mbrl.user_config import DATA_DIR, FORCE_DATESTAMP


def setup_logger_kwargs(
    exp_name: str,
    seed: Optional[int] = None,
    data_dir: Optional[Union[str, Path]] = None,
    datestamp: bool = False,
) -> dict:
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in
    ``user_config.py``.

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DATA_DIR`` in ``user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ""
    relpath = "".join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = "".join([hms_time, "-", exp_name, "-s", str(seed)])
        else:
            subfolder = "".join([exp_name, "-s", str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), exp_name=exp_name)
    return logger_kwargs
