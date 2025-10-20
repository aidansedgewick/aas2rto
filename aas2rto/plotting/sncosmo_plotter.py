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

import sncosmo

from aas2rto.target import Target
from aas2rto.plotting.default_plotter import DefaultLightcurvePlotter

logger = getLogger(__name__.split(".")[-1])


def plot_sncosmo_lightcurve(
    target: Target, t_ref: Time = None, return_plotter=False, **kwargs
) -> plt.Figure:
    t_ref = t_ref or Time.now()
    plotter = SncosmoLightcurvePlotter.plot(target, t_ref=t_ref, **kwargs)
    if plotter.target_has_models and (not plotter.models_plotted):
        logger.warning(f"{target.target_id} has models but none were plotted")
    if return_plotter:
        return plotter
    return plotter.fig


class SncosmoLightcurvePlotter(DefaultLightcurvePlotter):

    @classmethod
    def plot(cls, target: Target, t_ref: Time = None, **kwargs):
        t_ref = t_ref or Time.now()

        plotter = cls(t_ref=t_ref, **kwargs)
        plotter.plot_target(target)
        return plotter

    def __init__(
        self,
        t_ref: Time = None,
        figsize: tuple = None,
        forecast_days: float = 15.0,
        backcast_days: float = None,
        grid_dt: float = 0.5,
        model_name: str = "sncosmo_salt",
    ):
        super().__init__(t_ref=t_ref, figsize=figsize)
        self.forecast_days = forecast_days
        self.backcast_days = backcast_days
        self.grid_dt = grid_dt

        self.target_has_models = False
        self.models_plotted = False
        self.samples_plotted = False
        self.model_name = model_name

    def plot_target(self, target: Target):
        self.plot_sncosmo_models(target)
        super().plot_target(target)
        # self.plot_photometry(target)
        # self.add_cutouts(target)
        # self.format_axes(target)
        # self.add_comments(target)

    def plot_sncosmo_models(self, target: Target):
        model = target.models.get(self.model_name, None)
        if model is None:
            return
        if target.compiled_lightcurve is None:
            return
        self.target_has_models = True
        lightcurve = target.compiled_lightcurve
        t_start = target.compiled_lightcurve["mjd"].min()
        if self.backcast_days is not None:
            t_start = self.t_ref.mjd - self.backcast_days
        t_end = self.t_ref.mjd + self.forecast_days
        tgrid_main = np.arange(t_start, t_end + self.grid_dt, self.grid_dt)

        for ii, (band, band_history) in enumerate(lightcurve.groupby(self.band_col)):
            band_color = self.plot_colors.get(band, f"C{ii%8}")
            band_kwargs = dict(color=band_color)

            logger.debug(f"models for {target.target_id}")

            if self.tag_col in band_history.columns:
                detections = band_history[band_history[self.tag_col] == self.valid_tag]
            else:
                detections = band_history
            logger.debug(f"{len(detections)} detections band {band}")
            if len(detections) == 0:
                continue

            try:
                model_flux = model.bandflux(band, tgrid_main, zp=8.9, zpsys="ab")
            except ValueError as e:
                continue
            pos_mask = model_flux > 0.0
            if sum(pos_mask) == 0:
                logger.warning(f"{target.target_id} no pos flux for model {band}")
                continue
            model_flux = model_flux[pos_mask]
            tgrid = tgrid_main[pos_mask]

            tgrid_shift = tgrid - self.t_ref.mjd

            sample_tgrid_start = tgrid[0] - self.grid_dt
            sample_tgrid_end = tgrid[-1] + 1.5 * self.grid_dt
            samples_tgrid = np.arange(
                sample_tgrid_start, sample_tgrid_end, self.grid_dt
            )
            samples_tgrid_shift = samples_tgrid - self.t_ref.mjd

            model_mag = -2.5 * np.log10(model_flux) + 8.9
            self.peakmag_vals.append(np.nanmin(model_mag))  # axes fmt in parent class

            samples = getattr(model, "result", {}).get("samples", None)

            if samples is not None:
                ls = ":"  # plot the sample median with solid
            else:
                ls = "-"
            self.ax.plot(tgrid_shift, model_mag, color=band_color, ls=ls)
            self.models_plotted = True

            if samples is not None:
                median_model = get_model_median_params(model)

                model_med_flux = median_model.bandflux(band, tgrid, zp=8.9, zpsys="ab")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model_med_mag = -2.5 * np.log10(model_med_flux) + 8.9

                self.ax.plot(tgrid_shift, model_med_mag, color=band_color)

                # Now deal with samples.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    quartiles = [0.16, 0.5, 0.84]
                    samples_lb, samples_med, samples_ub = get_sample_quartiles(
                        samples_tgrid, model, band, q=quartiles
                    )
                    fb_kwargs = dict(color=band_color, alpha=0.2)
                    self.ax.fill_between(
                        samples_tgrid_shift, samples_lb, samples_ub, **fb_kwargs
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
            # extra [0] indexing because ax.plot returns LIST of n-1 lines for n points.
            self.ax.add_artist(lines_legend)
        return


def get_model_median_params(model: sncosmo.Model, burnin=0):
    model_copy = copy.deepcopy(model)
    vparam_names = model.result.get("vparam_names")
    samples = model.result.get("samples", None)
    if samples is None:
        msg = f"model result {model.get('result', {}).keys()} has no samples"
        raise ValueError(msg)

    if vparam_names is None:
        raise ValueError(f"model.result has no 'vparam_names'")

    median_params = np.nanquantile(samples, q=0.5, axis=0)

    pdict = {k: v for k, v in zip(vparam_names, median_params)}
    model_copy.update(pdict)
    return model_copy


def get_sample_quartiles(
    time_grid: Time, model: sncosmo.Model, band: str, q=0.5, spacing=20, burnin=0
):
    model_copy = copy.deepcopy(model)
    vparam_names = model.result.get("vparam_names")
    samples = model.result.get("samples", None)
    if samples is None:
        msg = f"model result {model.get('result', {}).keys()} has no samples"
        raise ValueError(msg)

    if vparam_names is None:
        raise ValueError(f"model.result has no 'vparam_names'")

    lc_evaluations = []
    t_start = time.perf_counter()
    for p_jj, params in enumerate(samples[burnin::spacing]):
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
