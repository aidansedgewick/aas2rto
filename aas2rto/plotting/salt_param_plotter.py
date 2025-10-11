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


x1_data = load_x1_datasets()
c_data = load_c_datasets()


class SaltParamPlotter:
    """This is actually pretty gross."""

    def __init__(self):
        self.plot_name = "salt_params"

    def __call__(self, target: Target, t_ref=None):
        return plot_salt_parameters(target)


def plot_salt_parameters(target: Target, t_ref=None) -> plt.Figure:
    salt_model = target.models.get("sncosmo_salt", None)
    if salt_model is None:
        return None
    plotter = SaltParamFigure.plot(target)
    return plotter.fig


class SaltParamFigure:

    default_figsize = (8, 4)

    def __init__(self):
        self.init_fig()
        self.lit_handles = []

        self.plot_literature()
        self.format_axes()

    @classmethod
    def plot(cls, target):
        plotter = cls()
        plotter.plot_target(target)
        return plotter

    def init_fig(self, figsize: tuple = None):
        figsize = figsize or self.default_figsize
        self.fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(1, 2)

        self.x1_ax = self.fig.add_subplot(gs[:, :1])
        self.c_ax = self.fig.add_subplot(gs[:, 1:])

    def plot_literature(self):
        for label, data in x1_data.items():
            self.x1_ax.step(data["param"], data["norm"])
        for label, data in c_data.items():
            lines = self.c_ax.step(data["param"], data["norm"], label=label)
            self.lit_handles.append(lines[0])

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

        self.x1_ax.axvline(x1, color="k", lw=2)
        self.c_ax.axvline(c, color="k", lw=2)

    def format_axes(self):
        self.x1_ax.set_xlabel("SALT x1")
        self.c_ax.set_xlabel("SALT c [mag]")

        legend_kwargs = dict(loc="upper center", fontsize=8, ncol=5)

        lit_legend = self.fig.legend(
            handles=self.lit_handles, bbox_to_anchor=(0.5, 0.98), **legend_kwargs
        )
        self.fig.add_artist(lit_legend)
