from __future__ import annotations

import shutil
import time
import warnings
from logging import getLogger
from typing import Any

from pathlib import Path

import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

logger = getLogger("aas2rto_utils")


def chunk_list(l, chunk_size=100):
    for ii in range(0, len(l), chunk_size):
        yield l[ii : ii + chunk_size]


def print_header(s: str) -> None:
    """
    Print a nicely formatted header line.

    Parameters
    ----------
    s [str]

    eg. if your string s="Hello!", print:\n
    `### ============ Hello! ============ ###`
    """
    try:
        tsize = shutil.get_terminal_size()
        cols = tsize.columns
    except Exception:
        cols = 60
    pad = (cols - (len(s) + 2) - 6) // 2

    pad = max(pad, 1)
    fmt_s = "\n###" + "=" * pad + f" {s} " + "=" * pad + "###"
    print(fmt_s)


def get_observatory_name(observatory: Observer | str | None):
    if isinstance(observatory, str):
        return observatory
    return observatory.name


def format_link_as_html(link: str, text: str = "", prefix="https://"):
    text = text or link
    return f"<a href='{prefix}{link}'>{text}</a>"


def format_link_as_markdown(link: str, text: str = "", prefix="https://"):
    if text:
        return f"<{prefix}{link}|{text}>"
    return f"{prefix}{link}"


def print_ascii_banner():

    s = """
    \033[32;1m                                \033[36;1m________  \033[32;1m               ___
    \033[32;1m                               \033[36;1m/  ___  _\\ \033[32;1m              /  / 
    \033[32;1m     _________________________\033[36;1m/  _/  \\  _\\\033[32;1m  ___________/ _/__   __________
    \033[32;1m    /_____  _/_____  _/ ______\033[36;1m\\__/ __/ __/\033[32;1m / ______/__  ____/  / _____  _/
    \033[32;1m   ______/ _/_____/ _/ _/____   \033[36;1m__/ ___/  \033[32;1m/ _/       / _/     / _/   / _/
    \033[32;1m  / ____  _/ ____  _/\\_____ _\\\033[36;1m_/ ___/ ___\033[32;1m/ _/       / _/  ___/ _/   / _/
    \033[32;1m / _/__/ _/ _/__/ _/______/ _\033[36;1m/  _/___/  /\033[32;1m _/       / _/__/ _/  _\\__/ _/
    \033[32;1m \\_______/\\_______/\\________/\033[36;1m\\_________/\033[32;1m__/        \\_______/\\________/
    \033[0m"""
    print(s)
