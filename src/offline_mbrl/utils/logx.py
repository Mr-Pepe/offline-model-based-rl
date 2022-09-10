#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=consider-using-f-string

"""Some simple logging functionality, inspired by rllab's logging.

Based on https://spinningup.openai.com
"""

import atexit
import json
import os
import os.path as osp
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return f"\x1b[{attr}m{string}\x1b[0m"


class EpochLogger:
    """A general-purpose logger for tracking values over epochs.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        output_filename: str = "progress.txt",
        experiment_name: str = None,
        env_name: str = "",
    ) -> None:
        """Initializes a Logger.

        Args:
            output_dir (string): A directory for saving results to. If ``None``,
                defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_filename (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.
            experiment_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``experiment_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``experiment_name``.)
            env_Name (str): The environment name used in the experiment.
        """
        self.output_dir = output_dir or f"/tmp/experiments/{time.time()}"
        if osp.exists(self.output_dir):
            print(
                "Warning: Log dir %s already exists! \
                Storing info there anyway."
                % self.output_dir
            )
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(  # pylint: disable=consider-using-with
            osp.join(self.output_dir, output_filename), "w", encoding="utf-8"
        )
        atexit.register(self.output_file.close)
        print(
            colorize("Logging data to %s" % self.output_file.name, "green", bold=True)
        )
        self.first_row = True
        self.log_headers: list[str] = []
        self.log_current_row: dict = {}
        self.exp_name = experiment_name
        self.env_name = env_name

        self.pytorch_saver_elements: dict[str, nn.Module] = {}

        tensorboard_path = os.path.join(self.output_dir, "tensorboard")
        shutil.rmtree(tensorboard_path, ignore_errors=True)
        self.tensorboard_writer = SummaryWriter(tensorboard_path, flush_secs=30)

        self.epoch_dict: dict[str, Any] = {}

    def store(self, **kwargs: dict) -> None:
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if k not in self.epoch_dict:
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(
        self,
        key: str,
        x_tick: int = 0,
        val: Any = None,
        with_min_and_max: bool = False,
        average_only: bool = False,
    ) -> None:
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        # pylint: disable=arguments-renamed
        plot_name = key

        for prefix in ["EpLen", "EpRet"]:
            if prefix in key:
                plot_name = "Performance/" + key

        for prefix in ["LossPi", "LossQ", "LogPi", "Vals"]:
            if prefix in key:
                plot_name = "Agent/" + key

        for prefix in ["EnvModel", "RolloutLength"]:
            if prefix in key:
                plot_name = "Model/" + key

        for prefix in ["Interacts", "Time"]:
            if prefix in key:
                plot_name = "RunMetrics/" + key

        scalars = {}

        if val is not None:
            self._log_tabular(key, val=val)
            scalars.update({key: val})
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            stats = get_statistics(vals)

            self._log_tabular(key if average_only else "Average" + key, val=stats[0])
            scalars.update({"Average" + key: stats[0]})

            if not average_only:
                self._log_tabular("Std" + key, val=stats[1])
            if with_min_and_max:
                self._log_tabular("Max" + key, val=stats[3])
                scalars.update({"Max" + key: stats[3]})
                self._log_tabular("Min" + key, val=stats[2])
                scalars.update({"Min" + key: stats[2]})

        if key != "Epoch":
            self.tensorboard_writer.add_scalars(plot_name, scalars, x_tick)

        self.epoch_dict[key] = []

    def _log_tabular(self, key: str, val: Any) -> None:
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key % s that you didn't include \
                 in the first iteration"
                % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. Maybe you forgot \
                to call dump_tabular()"
            % key
        )
        self.log_current_row[key] = val

    def get_stats(self, key: str) -> tuple[float, float, float, float]:
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = (
            np.concatenate(v)
            if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
            else v
        )
        return get_statistics(vals)

    def log(self, msg: str, color: str = "green") -> None:
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def save_config(self, config: dict) -> None:
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        if self.exp_name is not None:
            config["exp_name"] = self.exp_name
        output = json.dumps(config, separators=(",", ":\t"), indent=4, sort_keys=True)
        print(colorize("Saving config:\n", color="cyan", bold=True))
        print(output)
        with open(
            osp.join(self.output_dir, "config.json"), "w", encoding="utf-8"
        ) as out:
            out.write(output)

    def save_state(self, state_dict: dict, itr: Optional[int] = None) -> None:
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you
        previously set up saving for with ``setup_tf_saver``.

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        fname = "vars.pkl" if itr is None else "vars%d.pkl" % itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, fname))
        except Exception:  # pylint: disable=broad-except
            self.log("Warning: could not pickle state_dict.", color="red")
        if hasattr(self, "pytorch_saver_elements"):
            self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save: dict[str, nn.Module]) -> None:
        """
        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def add_to_pytorch_saver(self, what_to_save: dict[str, nn.Module]) -> None:
        self.pytorch_saver_elements.update(what_to_save)

    def _pytorch_simple_save(self, itr: Optional[int] = None) -> None:
        """Saves the PyTorch model (or models)."""
        assert hasattr(
            self, "pytorch_saver_elements"
        ), "First have to setup saving with self.setup_pytorch_saver"
        fpath = "pyt_save"
        fpath = osp.join(self.output_dir, fpath)

        for key, value in self.pytorch_saver_elements.items():
            fname = key + ("%d" % itr if itr is not None else "") + ".pt"
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.
                torch.save(value, fname)

    def dump_tabular(self) -> None:
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


def get_statistics(x: Iterable) -> tuple[float, float, float, float]:
    """
    Get mean/std and min/max of scalar x.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
    """
    x = np.array(x, dtype=np.float32)

    return x.mean(), x.std(), x.min(), x.max()
