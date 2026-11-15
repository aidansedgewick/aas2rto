from __future__ import annotations

import difflib
import warnings
from logging import getLogger
from typing import Iterable

from aas2rto.exc import (
    ConfigError,
    MissingKeysError,
    MissingKeysWarning,
    UnexpectedKeysError,
    UnexpectedKeysWarning,
)

logger = getLogger("config_utils")


def check_config_keys(
    provided, expected, name: str = None, warn: bool = True
) -> tuple[list, list]:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()
    unexpected_keys = check_unexpected_config_keys(
        provided, expected, name=name, warn=warn
    )
    missing_keys = check_missing_config_keys(provided, expected, name=name, warn=warn)
    return unexpected_keys, missing_keys


def check_unexpected_config_keys(
    provided: Iterable,
    expected: Iterable,
    name: str = None,
    warn: bool = True,
    raise_exc: bool = False,
    exc_class: type[ConfigError] = UnexpectedKeysError,
) -> list:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()

    unexpected_keys = set(provided) - set(expected)
    if len(unexpected_keys) > 0:
        msg = "\033[33;1munexpected keys\033[0m"
        if name is not None:
            msg = msg + f" in {name}"

        keys_str = ", ".join(f"\033[33;1m'{k}'\033[0m" for k in unexpected_keys)
        suggest_str = ""

        print_all_keys = False
        for k in sorted(unexpected_keys):
            suggestions = difflib.get_close_matches(k, expected, cutoff=0.7)
            if suggestions:
                if not suggest_str:
                    suggest_str = "Hints:"
                note = (
                    f"    for '{k}', did you mean "
                    + " or ".join(f"'{s}'" for s in suggestions)
                    + "?\n"
                )
                suggest_str = suggest_str + f"\n {note}"
            else:
                print_all_keys = True

        if print_all_keys:
            all_keys_note = "allowed keys:\n" + "\n".join(
                f"    {k}" for k in sorted(expected)
            )
            suggest_str = suggest_str + all_keys_note

        msg = msg + ":\n    " + keys_str + "\n" + suggest_str

        if raise_exc:  # Raise FIRST, so logs are
            logger.error(msg)
            raise exc_class(msg)
        if warn:
            logger.warning(msg)
            warnings.warn(UnexpectedKeysWarning(msg))
    return list(unexpected_keys)


def check_missing_config_keys(
    provided: Iterable,
    expected: Iterable,
    name: str = None,
    warn: bool = True,
    raise_exc: bool = False,
    exc_class: type[ConfigError] = MissingKeysError,
) -> list:

    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()

    missing_keys = set(expected) - set(provided)
    if missing_keys:
        msg = "\033[33;1mmissing keys\033[0m"
        if name is not None:
            msg = msg + f" in {name}"
        keys_str = ", ".join(f"\033[33;1m'{k}'\033[0m" for k in missing_keys)
        suggest_str = ""

        print_all_keys = False
        for k in sorted(missing_keys):
            suggestions = difflib.get_close_matches(k, provided, cutoff=0.7)
            if suggestions:
                if not suggest_str:
                    suggest_str = "Hints:"
                note = (
                    f"    for "
                    + " or ".join(f"'{s}'" for s in suggestions)
                    + f" did you mean '{k}'?\n"
                )
                suggest_str = suggest_str + f"\n {note}"
            else:
                print_all_keys = True

        if print_all_keys:
            all_keys_note = "required keys:\n" + "\n".join(
                f"    {k}" for k in sorted(expected)
            )
            suggest_str = suggest_str + all_keys_note

        msg = msg + ":\n    " + keys_str + "\n" + suggest_str
        if raise_exc:
            logger.error(msg)
            raise exc_class(msg)
        if warn:
            logger.warning(msg)
            warnings.warn(MissingKeysWarning(msg))
    return list(missing_keys)
