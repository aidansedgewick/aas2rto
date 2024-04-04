import copy
import time
import warnings
from logging import getLogger

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.interpolate import CubicSpline

from astropy import units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from dk154_targets.target import Target
from dk154_targets.plotters.default_plotter import DefaultLightcurvePlotter

logger = getLogger(__name__.split(".")[-1])


def plot_sncosmo_lightcurve(target: Target, t_ref: Time = None) -> plt.Figure:
    t_ref = t_ref or Time.now()
    plotter = SncosmoLightcurvePlotter.plot(target, t_ref=t_ref)
    if plotter.target_has_models and (not plotter.models_plotted):
        logger.warning(f"{target.objectId} has models but none were plotted")
    return plotter.fig


class SncosmoLightcurvePlotter(DefaultLightcurvePlotter):

    @classmethod
    def plot(cls, target: Target, t_ref: Time = None, **kwargs) -> plt.Figure:
        t_ref = t_ref or Time.now()

        plotter = cls(t_ref=t_ref, **kwargs)
        plotter.plot_sncosmo_models(target)
        plotter.plot_photometry(target)
        plotter.add_cutouts(target)
        plotter.format_axes(target)
        plotter.add_comments(target)
        return plotter

    def __init__(
        self, t_ref: Time = None, figsize=None, forecast_days=15.0, grid_dt=0.2
    ):
        super().__init__(t_ref=t_ref, figsize=figsize)
        self.forecast_days = forecast_days
        self.grid_dt = grid_dt

        self.models_plotted = False
        self.samples_plotted = False

    def plot_sncosmo_models(self, target: Target):

        model = target.models.get("sncosmo_salt", None)
        if model is None:
            return
        if target.compiled_lightcurve is None:
            return
        lightcurve = target.compiled_lightcurve
        t_start = target.compiled_lightcurve["jd"].min()
        t_end = self.t_ref.jd + self.forecast_days
        tgrid_main = np.arange(t_start, t_end + self.grid_dt, self.grid_dt)

        for ii, (band, band_history) in enumerate(lightcurve.groupby(self.band_col)):
            band_color = self.plot_colors.get(band, f"C{ii%8}")
            band_kwargs = dict(color=band_color)

            logger.debug(f"models for {target.objectId}")

            if self.tag_col in band_history.columns:
                detections = band_history[band_history[self.tag_col] == self.valid_tag]
            else:
                detections = band_history
            logger.debug(f"{len(detections)} detections band {band}")
            if len(detections) == 0:
                continue

            model_flux = model.bandflux(band, tgrid_main, zp=8.9, zpsys="ab")
            pos_mask = model_flux > 0.0
            if sum(pos_mask) == 0:
                logger.warning(f"{target.objectId} no pos flux for model {band}")
                continue
            model_flux = model_flux[pos_mask]
            tgrid = tgrid_main[pos_mask]

            tgrid_shift = tgrid - self.t_ref.jd

            sample_tgrid_start = tgrid[0] - self.grid_dt
            sample_tgrid_end = (tgrid[-1] + 1.5 * self.grid_dt,)
            samples_tgrid = np.arange(
                sample_tgrid_start, sample_tgrid_end, self.grid_dt
            )
            samples_tgrid_shift = samples_tgrid - self.t_ref.jd

            model_mag = -2.5 * np.log10(model_flux) + 8.9
            self.peakmag_vals.append(np.nanmin(model_mag))

            samples = getattr(model, "result", {}).get("samples", None)

            if samples is not None:
                ls = ":"
            else:
                ls = "-"
            self.ax.plot(tgrid_shift, model_mag, color=band_color, ls=ls)
            self.models_plotted = True

            if samples is not None:
                vparam_names = model.result.get("vparam_names")
                median_model = get_model_median_params(model, vparam_names=vparam_names)

                model_med_flux = median_model.bandflux(band, tgrid, zp=8.9, zpsys="ab")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model_med_mag = -2.5 * np.log10(model_med_flux) + 8.9

                self.ax.plot(tgrid_shift, model_med_mag, color=band_color)

                # Now deal with samples.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    samples_lb, samples_med, samples_ub = get_sample_quartiles(
                        samples_tgrid,
                        model,
                        band,
                        q=[0.16, 0.5, 0.84],
                        vparam_names=vparam_names,
                    )
                    self.ax.fill_between(
                        samples_tgrid_shift,
                        samples_lb,
                        samples_ub,
                        color=band_color,
                        alpha=0.2,
                    )
                    self.samples_plotted = True
                    self.ax.plot(
                        samples_tgrid_shift, samples_med, color=band_color, ls="--"
                    )

        if self.samples_plotted:
            x, y = [0, 0], [23, 23]
            l0 = self.ax.plot(x, y, ls="-", color="k", label="median parameters")
            l1 = self.ax.plot(x, y, ls="--", color="k", label="LC samples median")
            l2 = self.ax.plot(x, y, ls=":", color="k", label="mean parameters")
            lines_legend = self.ax.legend(handles=[l0[0], l1[0], l2[0]], loc=4)
            # extra [0] indexing because ax.plot reutrns LIST of n-1 lines for n points.
            self.ax.add_artist(lines_legend)
        return


def get_model_median_params(model, vparam_names=None, samples=None):
    model_copy = copy.deepcopy(model)
    vparam_names = vparam_names or model.result.get("vparam_names")
    samples = samples or model.result.get("samples", None)
    if samples is None:
        msg = f"model result {model.get('result', {}).keys()} has no samples"
        raise ValueError(msg)

    median_params = np.nanquantile(samples, q=0.5, axis=0)

    pdict = {k: v for k, v in zip(vparam_names, median_params)}
    model_copy.update(pdict)
    return model_copy


def get_sample_quartiles(
    time_grid, model, band, samples=None, vparam_names=None, q=0.5, spacing=20
):
    model_copy = copy.deepcopy(model)
    vparam_names = vparam_names or model.result.get("vparam_names")
    samples = samples or model.result.get("samples", None)
    if samples is None:
        msg = f"model result {model.get('result', {}).keys()} has no samples"
        raise ValueError(msg)

    lc_evaluations = []
    t_start = time.perf_counter()
    for p_jj, params in enumerate(samples[::spacing]):
        pdict = {k: v for k, v in zip(vparam_names, params)}
        model_copy.update(pdict)
        lc_flux_jj = model_copy.bandflux(band, time_grid, zp=8.9, zpsys="ab")
        with np.errstate(divide="ignore", invalid="ignore"):
            lc_mag_jj = -2.5 * np.log10(lc_flux_jj) + 8.9
        lc_evaluations.append(lc_mag_jj)
    t_end = time.perf_counter()

    lc_evaluations = np.vstack(lc_evaluations)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    lc_bounds = np.nanquantile(lc_evaluations, q=q, axis=0)
    return lc_bounds


def spline_and_sample(x, y, xnew):
    ypos_mask = np.isfinite(y)
    xpos = x[ypos_mask]
    ypos = y[ypos_mask]
    spl = CubicSpline(xpos, ypos)
    xprime_mask = (xpos[0] < xnew) & (xnew < xpos[-1])
    xprime = xnew[xprime_mask]
    yprime = spl(xprime[xprime_mask])
    return xprime, yprime
