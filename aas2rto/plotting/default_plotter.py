import copy
import time
import traceback
import warnings
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import Table, vstack
from astropy.table import unique as unique_table
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from astroplan import FixedTarget, Observer
from astroplan.plots import plot_altitude

from aas2rto.exc import MissingDateError, UnknownObservatoryWarning
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.target import Target, DEFAULT_ZTF_BROKER_PRIORITY

logger = getLogger(__name__.split(".")[-1])

matplotlib.use("Agg")


def plot_default_lightcurve(target: Target, t_ref: Time = None) -> plt.Figure:
    t_ref = t_ref or Time.now()
    plotter = DefaultLightcurvePlotter.plot(target, t_ref=t_ref)
    return plotter.fig


class DefaultLightcurvePlotter:

    @classmethod
    def plot(cls, target: Target, t_ref: Time = None, **kwargs) -> plt.Figure:
        t_ref = t_ref or Time.now()

        plotter = cls(t_ref=t_ref, **kwargs)
        plotter.plot_target(target)
        return plotter

    def __init__(self, t_ref: Time = None, figsize: tuple = None):
        self.t_ref = t_ref or Time.now()

        self.set_default_plot_params()

        self.init_fig(figsize=figsize)
        self.legend_handles = []
        self.peakmag_vals = []
        self.faintmag_vals = []
        self.photometry_plotted = False
        self.cutouts_added = False
        self.axes_formatted = False
        self.comments_added = False

    def plot_target(self, target: Target):

        self.plot_photometry(target)
        self.add_cutouts(target)
        self.format_axes(target)
        self.add_comments(target)

    def set_default_plot_params(self):
        self.lc_gs = plt.GridSpec(3, 4)
        self.zscaler = ZScaleInterval()

        self.default_figsize = (6.5, 5)

        self.ztf_colors = {"ztfg": "C2", "ztfr": "C3", "ztfi": "C4"}
        self.atlas_colors = {"atlasc": "C9", "atlaso": "C1"}  # cyan, orange
        self.yse_colors = {
            f"ps1::{b}": f"C{ii}" for ii, b in enumerate("g r i z y w".split(), 2)
        }
        self.lsst_colors = {
            f"lsst{b}": f"C{ii}" for ii, b in enumerate("g r i z y w".split())
        }
        self.plot_colors = {
            **self.ztf_colors,
            **self.atlas_colors,
            **self.lsst_colors,
            **self.yse_colors,
            "no_band": "k",
        }

        ztf_shapes = {x: "o" for x in "ztfg ztfr ztfi".split()}
        atlas_shapes = {x: "o" for x in "atlasc atlaso".split()}
        yse_shapes = {f"ps1::{x}": "s" for x in "g r i z y w".split()}
        swift_shapes = {f"uvot::{x}": "v" for x in "u b v uvw1 uvw2 uvm1 uvm2".split()}
        self.plot_shapes = {**ztf_shapes, **atlas_shapes, **yse_shapes}

        self.valid_kwargs = dict(ls="none")  # , marker="o")
        self.ulimit_kwargs = dict(ls="none")  # , marker="v", mfc="none")
        self.badqual_kwargs = dict(ls="none", mfc="none")  # , marker="o")

        self.tag_col = "tag"
        self.valid_tag = "valid"
        self.ulimit_tag = "upperlim"
        self.badqual_tag = "badqual"

        self.mag_col = "mag"
        self.magerr_col = "magerr"
        self.diffmaglim_col = "diffmaglim"

        self.band_col = "band"

    def init_fig(self, figsize: tuple = None):
        figsize = figsize or self.default_figsize
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(self.lc_gs[:, :-1])

    def plot_photometry(self, target: Target, band_col=None):
        target_id = target.target_id

        if target.compiled_lightcurve is None:

            logger.warning(f"{target_id} has no compiled lightcurve for plotting.")
            return self.fig
        lightcurve = target.compiled_lightcurve.copy()

        if "mjd" not in lightcurve.columns:
            if "jd" in lightcurve.columns:
                time_dat = Time(lightcurve["jd"].values, format="jd")
            else:
                msg = f"{target_id} missing date column to plot lightcurve: {lightcurve.columns}"
                logger.error(msg)
                raise ValueError(msg)
            lightcurve.loc[:, "mjd"] = time_dat.mjd

        band_col = band_col or self.band_col
        if band_col not in lightcurve.columns:
            msg = (
                f"{target_id} has no column '{band_col}' in compiled_lightcurve.columns"
            )
            print(list(lightcurve.columns))
            logger.error(msg)
            lightcurve.loc[:, band_col] = "no_band"

        for ii, (band, band_history) in enumerate(lightcurve.groupby(band_col)):
            band_color = self.plot_colors.get(band, f"C{ii%8}")
            shape = self.plot_shapes.get(band, "o")
            det_kwargs = dict(color=band_color, marker=shape)
            lim_kwargs = dict(color=band_color, marker="$\u2193$")

            scatter_handle = self.ax.errorbar(
                0, 0, yerr=0.1, label=band, **det_kwargs, **self.valid_kwargs
            )
            self.legend_handles.append(scatter_handle)

            if self.tag_col in band_history.columns:
                detections = band_history[band_history[self.tag_col] == self.valid_tag]
                ulimits = band_history[band_history[self.tag_col] == self.ulimit_tag]
                badqual = band_history[band_history[self.tag_col] == self.badqual_tag]

                if len(ulimits) > 0:
                    xdat = ulimits["mjd"].values - self.t_ref.mjd
                    ydat = ulimits[self.diffmaglim_col]
                    self.ax.errorbar(xdat, ydat, **lim_kwargs, **self.ulimit_kwargs)
                if len(badqual) > 0:
                    xdat = badqual["mjd"].values - self.t_ref.mjd
                    ydat = badqual[self.mag_col].values
                    yerr = badqual[self.magerr_col].values
                    self.ax.errorbar(
                        xdat, ydat, yerr=yerr, **det_kwargs, **self.badqual_kwargs
                    )
                N_det = len(detections)
                N_badqual = len(badqual)
                N_ulim = len(ulimits)
                if N_det + N_badqual + N_ulim != len(band_history):
                    msg = (
                        f"{target_id}: N_det+N_badqual+N_ulim != len {band} lc "
                        + f"({N_det}+{N_badqual}+{N_ulim} != {len(band_history)})"
                    )
                    logger.warning("\n    " + msg)
            else:
                detections = band_history

            if len(detections) > 0:
                xdat = detections["mjd"] - self.t_ref.mjd
                ydat = detections[self.mag_col]
                yerr = detections[self.magerr_col]
                self.ax.errorbar(
                    xdat, ydat, yerr=yerr, **det_kwargs, **self.valid_kwargs
                )
                self.photometry_plotted = True
                self.peakmag_vals.append(ydat.min())
                self.faintmag_vals.append(ydat.max())

    def add_cutouts(self, target: Target):
        cutouts = {}
        for broker in DEFAULT_ZTF_BROKER_PRIORITY:
            source_data = target.target_data.get(broker, None)
            if source_data is None:
                continue
            if len(source_data.cutouts) == 0:
                continue
            cutouts = source_data.cutouts
            break

        for ii, imtype in enumerate(["Science", "Template", "Difference"]):
            im_ax = self.fig.add_subplot(self.lc_gs[ii : ii + 1, -1:])

            im_ax.set_xticks([])
            im_ax.set_yticks([])

            imtext_kwargs = dict(
                rotation=90,
                transform=im_ax.transAxes,
                ha="left",
                va="center",
                fontsize=12,
            )
            im_ax.text(1.08, 0.5, imtype, **imtext_kwargs)

            im = cutouts.get(imtype.lower(), None)
            if im is None:
                continue

            im_finite = im[np.isfinite(im)]
            vmin, vmax = self.zscaler.get_limits(im_finite.flatten())
            im_ax.imshow(im, vmin=vmin, vmax=vmax)

            xl_im = len(im.T)
            yl_im = len(im)
            # add crosshairs
            im_ax.plot(
                [0.5 * xl_im, 0.5 * xl_im], [0.2 * yl_im, 0.4 * yl_im], color="r"
            )
            im_ax.plot(
                [0.2 * yl_im, 0.4 * yl_im], [0.5 * yl_im, 0.5 * yl_im], color="r"
            )

            self.cutouts_added = True

    def format_axes(self, target: Target):

        self.fig.subplots_adjust(top=0.85)

        names = list(set(target.alt_ids.values()))
        title = " -- ".join(sorted(names))

        transform_kwargs = dict(ha="center", va="top", transform=self.fig.transFigure)
        self.ax.text(0.5, 0.98, title, fontsize=14, **transform_kwargs)

        t_ra = target.coord.ra
        t_dec = target.coord.dec
        subtitle = f"ra,dec=({t_ra.deg:.4f},{t_dec.deg:+.6f})"

        tns_data = target.target_data.get("tns", None)
        if tns_data is not None:
            known_redshift = float(tns_data.parameters.get("redshift", "nan"))
            if np.isfinite(known_redshift):
                subtitle = subtitle + r" ($z_{\rm TNS}=" + f"{known_redshift}" + "$)"

        self.ax.text(0.5, 0.93, subtitle, fontsize=11, **transform_kwargs)

        self.peakmag_vals.append(17.0)
        y_bright = np.nanmin(self.peakmag_vals) - 0.2
        self.faintmag_vals.append(22.0)
        y_faint = np.nanmax(self.faintmag_vals) + 0.2
        self.ax.set_ylim(y_faint, y_bright)
        self.ax.axvline(0, color="k")

        legend = self.ax.legend(handles=self.legend_handles, loc=2)
        self.ax.add_artist(legend)
        date_str = self.t_ref.strftime("%d-%b-%y %H:%M")
        xlabel = f"Days before {date_str}"
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel("Difference magnitude", fontsize=12)
        try:
            self.add_readable_dateticks()
        except Exception as e:
            pass
        self.axes_formatted = True

    def add_readable_dateticks(self):
        twiny = self.ax.twiny()
        twiny.set_xlim(self.ax.get_xlim())

        x0, x1 = self.ax.get_xlim()
        s = 10
        xmin = np.sign(x0) * np.floor(abs(x0) / s) * s
        xmax = np.sign(x1) * np.ceil(abs(x1) / s) * s
        xticks = self.t_ref.mjd + np.arange(xmin, xmax, s)
        xticklabels = [Time(int(x), format="mjd").strftime("%d %b") for x in xticks]
        twiny.set_xticks(xticks - self.t_ref.mjd)
        twiny.set_xticklabels(xticklabels)

    def add_comments(self, target):
        comments = target.score_comments.get("no_observatory", [])
        self.fig.subplots_adjust(bottom=0.3)
        if len(comments) > 0:
            N = len(comments) // 2
            text_col1 = "score comments:\n" + "\n".join(
                f"    {comm}" for comm in comments[:N]
            )
            transform_kwargs = dict(ha="left", va="top", transform=self.fig.transFigure)
            self.fig.text(0.03, 0.2, text_col1, fontsize=8, **transform_kwargs)
            text_col2 = "\n" + "\n".join(f"    {comm}" for comm in comments[N:])
            self.fig.text(0.53, 0.2, text_col2, fontsize=8, **transform_kwargs)
            self.comments_added = True
