#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains a function to convert boolean string values to their boolean value."""

import argparse


def str2bool(values: str) -> bool:
    """Converts a string to a boolean value.

    Args:
        value (str): The string value.

    Raises:
        argparse.ArgumentTypeError: If the string value was not recognized as
            representing a boolean.

    Returns:
        bool: The resulting boolean value.
    """
    if isinstance(values, bool):
        return values

    if values.lower() in ("yes", "true", "t", "y", "1"):
        return True

    if values.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")
