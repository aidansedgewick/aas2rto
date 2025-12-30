from logging import getLogger
from pathlib import Path

import numpy as np

import pandas as pd

import sncosmo

import matplotlib.pyplot as plt

from aas2rto.target import Target

logger = getLogger(__name__.split(".")[-1])


litdat_path = Path(__file__).parent / "litdat"


def load_x1_datasets():
    filename_lookup = {
        "FinkTNS": "FinkTNS_x1.csv",
        "Nielsen16 (JLA)": "Nielsen16_JLA_x1.csv",
        "Taylor21 (DES)": "Taylor21_DES_x1.csv",
        "Scolnic22 (P+)": "Scolnic22_Pantheon_x1.csv",
    }

    data = {}
    for label, filename in filename_lookup.items():
        filepath = litdat_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath, sep=r"\s+")
            data[label] = df
        else:
            logger.warning(f"NO '{label}' x1 data found at:\n   {filepath}")
    return data


def load_c_datasets():
    filename_lookup = {
        "FinkTNS": "FinkTNS_c.csv",
        "Nielsen16 (JLA)": "Nielsen16_JLA_c.csv",
        "Taylor21 (DES)": "Taylor21_DES_c.csv",
        "Scolnic22 (P+)": "Scolnic22_Pantheon_c.csv",
    }

    data = {}
    for label, filename in filename_lookup.items():
        filepath = litdat_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath, sep=r"\s+")
            data[label] = df
        else:
            logger.warning(f"NO '{label}' c data found at:\n   {filepath}")

    return data


class SaltParamPlottingWrapper:
    """This is actually pretty gross."""

    # Why is the plotting helper also a class here?
    # Normally it'd just be a function - ie. plot_salt_parameters()
    # But it's also helpful to have the plotter have other attrs we can ask for/change.
    # Why not just have the PlottingUtil class (below) as a function plot_
    #   - it make it easier to test individual steps of plotting, rather than
    #     one humongous function.

    def __init__(self, x1_data=None, c_data=None):
        self.plot_name = "salt_params"
        self.literature = None
        self.x1_data = x1_data or load_x1_datasets()
        self.c_data = c_data or load_c_datasets()

    def __call__(self, target: Target, t_ref=None, return_plotter=False):
        plotter = SaltParamPlotter.plot(
            target, x1_data=self.x1_data, c_data=self.c_data
        )
        if return_plotter:
            return plotter
        return plotter.fig


class SaltParamPlotter:

    default_figsize = (8, 4)

    def __init__(self):
        self.init_fig()
        self.lit_handles = []

        self.literature_plotted = False
        self.axes_formatted = False
        self.target_plotted = False
        self.intervals_plotted = False

    @classmethod
    def plot(cls, target: Target, x1_data: dict = None, c_data: dict = None):
        plotter = cls()
        plotter.plot_literature(x1_data=x1_data, c_data=c_data)
        plotter.plot_target(target)
        plotter.format_axes()
        return plotter

    def init_fig(self, figsize: tuple = None):
        figsize = figsize or self.default_figsize
        self.fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(1, 2)

        self.x1_ax = self.fig.add_subplot(gs[:, :1])
        self.c_ax = self.fig.add_subplot(gs[:, 1:])

    def plot_literature(self, x1_data=None, c_data=None):
        if x1_data is None:
            logger.warning("No `x1_data` provided for salt param plots! Load default.")
            x1_data = load_x1_datasets()
        if c_data is None:
            logger.warning("No `c_data` provided for salt param plots! Load default.")
            c_data = load_c_datasets()

        for label, data in x1_data.items():
            self.x1_ax.step(data["param"], data["norm"])
            self.literature_plotted = True  # mainly for unit tests
        for label, data in c_data.items():
            lines = self.c_ax.step(data["param"], data["norm"], label=label)
            self.lit_handles.append(lines[0])
            self.literature_plotted = True  # mainly for unit tests

    def plot_target(self, target):
        salt_model = target.models.get("sncosmo_salt", None)
        if salt_model is None:
            return None

        x1 = salt_model["x1"]
        c = salt_model["c"]

        samples = salt_model.result.get("samples", None)
        vparam_names = salt_model.result.get("vparam_names", None)
        if samples is not None and vparam_names is not None:
            param_errs = np.std(samples, axis=0)
            err_dict = {p: x for p, x in zip(vparam_names, param_errs)}

            x1_err = err_dict["x1"]
            c_err = err_dict["c"]

            self.x1_ax.axvspan(x1 - x1_err, x1 + x1_err, color="k", alpha=0.3)
            self.c_ax.axvspan(c - c_err, c + c_err, color="k", alpha=0.3)
            self.intervals_plotted = True

        self.x1_ax.axvline(x1, color="k", lw=2)
        self.c_ax.axvline(c, color="k", lw=2)
        self.target_plotted = True  # mainly for unit tests

    def format_axes(self):
        self.x1_ax.set_xlabel("SALT x1")
        self.c_ax.set_xlabel("SALT c [mag]")

        legend_kwargs = dict(loc="upper center", fontsize=8, ncol=5)

        lit_legend = self.fig.legend(
            handles=self.lit_handles, bbox_to_anchor=(0.5, 0.98), **legend_kwargs
        )
        self.fig.add_artist(lit_legend)
        self.axes_formatted = True  # mainly for unit tests
