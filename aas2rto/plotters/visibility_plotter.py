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

from aas2rto.exc import MissingDateError, UnknownObservatoryWarning
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.target import Target
from aas2rto.utils import get_observatory_name

logger = getLogger(__name__.split(".")[-1])

matplotlib.use("Agg")


def plot_visibility(
    observatory: Observer, *target_list: Target, return_plotter=False, **kwargs
):
    plotter = VisibilityPlotter.plot(observatory, *target_list, **kwargs)
    if return_plotter:
        return plotter
    return plotter.fig


class VisibilityPlotter:
    double_height_figsize = (6, 8)
    single_height_figsize = (6, 4)

    moon_kwargs = {"color": "0.5", "ls": "--", "label": "moon"}
    sun_kwargs = {"color": "0.5", "label": "sun"}

    title_kwargs = {"fontsize": 14, "ha": "center", "va": "top"}

    # az_ticks = np.arange(0, 2 * np.pi, np.pi / 4)
    az_ticklabels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    @classmethod
    def plot(
        cls,
        observatory: Observer,
        *target_list: Target,
        alt_ax=True,
        sky_ax=True,
        sun=True,
        moon=True,
        obs_info: ObservatoryInfo = None,
        recompute=False,
        t_ref: Time = None,
        t_grid: Time = None,
        dt: float = None,
        forecast: float = None,
        figsize: tuple = None,
    ):
        plotter = cls(
            observatory,
            obs_info=obs_info,
            t_ref=t_ref,
            t_grid=t_grid,
            alt_ax=alt_ax,
            sky_ax=sky_ax,
            dt=dt,
            forecast=forecast,
            figsize=figsize,
        )

        for ii, target in enumerate(target_list):
            plotter.plot_target(target, recompute=recompute, idx=ii)
        if sun:
            plotter.plot_sun()
        if moon:
            plotter.plot_moon()
        plotter.format_axes()
        return plotter

    def __init__(
        self,
        observatory: Observer,
        obs_info: ObservatoryInfo = None,
        t_ref: Time = None,
        t_grid: Time = None,
        alt_ax: bool = True,
        sky_ax: bool = True,
        dt: float = None,
        forecast: float = None,
        figsize: tuple = None,
    ):
        t_ref = t_ref or Time.now()

        self.observatory = observatory

        if obs_info is None:
            logger.info(f"recomputing obs_info for {observatory.name}")
            obs_info = ObservatoryInfo.for_observatory(
                self.observatory,
                t_grid=t_grid,
                t_ref=t_ref,
                dt=dt,
                forecast=forecast,
            )
        self.obs_info = obs_info
        self.t_grid = self.obs_info.t_grid

        self.init_fig(alt_ax=alt_ax, sky_ax=sky_ax, figsize=figsize)
        self.init_axes(alt_ax=alt_ax, sky_ax=sky_ax)
        self.altitude_plotted = False
        self.sky_plotted = False
        self.legend_handles = []
        self.sky_plotted = False
        self.moon_plotted = False
        self.sun_plotted = False
        self.axes_formatted = False

    def init_fig(self, alt_ax=True, sky_ax=True, figsize=None):
        if alt_ax and sky_ax:
            figsize = figsize or self.double_height_figsize
        else:
            figsize = figsize or self.single_height_figsize
        self.fig = plt.figure(figsize=figsize)
        self.alt_ax = None
        self.sky_ax = None

    def init_axes(self, alt_ax=True, sky_ax=True):
        if alt_ax and sky_ax:
            self.alt_ax = self.fig.add_axes(211)
            self.sky_ax = self.fig.add_axes(212, projection="polar")
        if alt_ax and (not sky_ax):
            self.alt_ax = self.fig.add_axes(111)
        if (not alt_ax) and sky_ax:
            self.sky_ax = self.fig.add_axes(111, projection="polar")

    def plot_target(self, target: Target, idx: int = None, recompute=False, **kwargs):
        obs_name = get_observatory_name(self.observatory)
        if isinstance(target, Target):
            coord = target.coord
            target_label = target.objectId
            obs_info = target.observatory_info.get(obs_name, None)
        elif isinstance(target, SkyCoord):
            coord = target
            target_label = None
            if idx is not None:
                target_label = f"target_{idx+1:02d}"
            obs_info = None
        else:
            raise TypeError(
                f"target should be `Target` or `SkyCoord`, not {type(target)}"
            )
        if obs_info is None or recompute:
            logger.info(f"recompute altaz for {target_label}")
            altaz = self.observatory.altaz(self.t_grid, coord)
        else:
            altaz = obs_info.target_altaz

        plot_kwargs = dict(
            label=target_label, color=f"C{idx%8}", ls=("-" if idx < 8 else "--")
        )
        plot_kwargs.update(kwargs)
        if self.alt_ax is not None:
            self.plot_altitude(self.t_grid, altaz, **plot_kwargs)
            self.altitude_plotted = True
        if self.sky_ax is not None:
            self.plot_sky(self.t_grid, altaz, **plot_kwargs)
            self.sky_plotted = True
        return

    def plot_altitude(self, t_grid: Time, altaz: AltAz, **kwargs):
        if self.alt_ax is None:
            msg = "alt_ax is None. either `plotter.init_axes(alt_ax=True)` or set plotter.alt_ax"
            raise ValueError(msg)
        line = self.alt_ax.plot(t_grid.mjd, altaz.alt.deg, **kwargs)
        self.legend_handles.append(line[0])
        return

    def plot_sky(self, t_grid, altaz: AltAz, **kwargs):
        if self.sky_ax is None:
            msg = "alt_ax is None. either `plotter.init_axes(alt_ax=True)` or set plotter.sky_ax"
            raise ValueError(msg)
        line = self.sky_ax.plot(altaz.az.rad, altaz.alt.deg, **kwargs)
        if self.alt_ax is None:
            self.legend_handles.append(line[0])
        return

    def plot_moon(self, **kwargs):
        moon_altaz = self.obs_info.moon_altaz
        if moon_altaz is None:
            moon_altaz = self.observatory.moon_altaz(self.t_grid)
        moon_kwargs = self.moon_kwargs.copy()
        moon_kwargs.update(kwargs)
        if self.alt_ax is not None:
            self.plot_altitude(self.t_grid, moon_altaz, **moon_kwargs)
        if self.sky_ax is not None:
            self.plot_sky(self.t_grid, moon_altaz, **moon_kwargs)
        self.moon_plotted = True
        return

    def plot_sun(self, **kwargs):
        sun_altaz = self.obs_info.sun_altaz
        if sun_altaz is None:
            sun_altaz = self.observatory.sun_altaz(self.t_grid)
        sun_kwargs = self.sun_kwargs.copy()
        sun_kwargs.update(kwargs)
        if self.alt_ax is not None:
            self.plot_altitude(self.t_grid, sun_altaz, **sun_kwargs)
        if self.sky_ax is not None:
            self.plot_sky(self.t_grid, sun_altaz, **sun_kwargs)
        self.sun_plotted = True
        return

    def format_axes(self):
        obs_name = self.observatory.name
        if obs_name is None:
            loc = self.observatory.location
            obs_name = f"{str(loc.lat.dms)}, {str(loc.lon.dms)}"
        t_str = self.t_grid[0].strftime("%Y-%m-%d %H:%M") + " UT"
        title = f"Observing from {obs_name} ({t_str})"
        self.fig.text(
            0.5, 0.98, title, transform=self.fig.transFigure, **self.title_kwargs
        )
        if self.alt_ax is not None:
            self.alt_ax.set_xlim(self.t_grid[0].mjd, self.t_grid[-1].mjd)
            self.alt_ax.set_ylim(-20.0, 90.0)
            self.alt_ax.axhline(0.0, color="k")
            self.alt_ax.set_ylabel(f"Altitude [deg]")
            try:
                self.set_readable_alt_xticks()
            except Exception as e:
                pass
            legend = self.alt_ax.legend(handles=self.legend_handles, ncols=3)
            self.alt_ax.add_artist(legend)
        if self.sky_ax is not None:
            self.sky_ax.set_rlim(bottom=90.0, top=0.0)
            self.sky_ax.set_rgrids((15.0, 30.0, 45.0, 60.0, 75.0))
            self.sky_ax.set_theta_zero_location("N")
            az_ticks = self.sky_ax.get_xticks()
            self.sky_ax.set_xticks(az_ticks)
            self.sky_ax.set_xticklabels(self.az_ticklabels)
        self.axes_formatted = True

    def set_readable_alt_xticks(self):
        n = 8
        delta = int(24 * (self.t_grid[-1].mjd - self.t_grid[0].mjd) / n)
        if delta < 1:
            return
        d = self.t_grid[0].ymdhms
        t0 = Time(dict(year=d.year, month=d.month, day=d.day, hour=int(d.hour)))
        vals = t0 + np.arange(1, n * delta, delta) * u.hour  # start=1 as 0 in past..
        ticks = vals[vals.mjd < self.t_grid[-1].mjd]
        tk0 = ticks[0].strftime("%H:%M\n%b %d")
        labels = [tk0]
        for tk in ticks[1:]:
            if tk.ymdhms.hour < delta:
                labels.append(tk.strftime("%H:%M\n%b %d"))
            else:
                labels.append(tk.strftime("%H:%M"))
        self.alt_ax.set_xticks(ticks.mjd)
        self.alt_ax.set_xticklabels(labels)
