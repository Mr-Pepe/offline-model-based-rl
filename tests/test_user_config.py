#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""This module contains unit tests for setting the user configuration."""

import importlib

from pytest import MonkeyPatch

# pylint: disable=import-outside-toplevel


def test_override_data_dir_with_env_var(monkeypatch: MonkeyPatch) -> None:
    import offline_mbrl.user_config as config

    assert config.DATA_DIR != "data_dir_test"

    monkeypatch.setenv("OMBRL_DATA_DIR", "data_dir_test")

    importlib.reload(config)

    assert config.DATA_DIR == "data_dir_test"


def test_override_models_dir_with_env_var(monkeypatch: MonkeyPatch) -> None:
    import offline_mbrl.user_config as config

    assert config.MODELS_DIR != "models_dir_test"

    monkeypatch.setenv("OMBRL_MODELS_DIR", "models_dir_test")

    importlib.reload(config)

    assert config.MODELS_DIR == "models_dir_test"


def test_override_force_datestamp_with_env_var(monkeypatch: MonkeyPatch) -> None:
    import offline_mbrl.user_config as config

    assert isinstance(config.FORCE_DATESTAMP, bool)

    monkeypatch.setenv("OMBRL_FORCE_DATESTAMP", "True")

    importlib.reload(config)

    assert isinstance(config.FORCE_DATESTAMP, bool)
    assert config.FORCE_DATESTAMP is True
