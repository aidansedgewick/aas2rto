import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from astroplan import FixedTarget, Observer

from dk154_targets.exc import (
    MissingDateError,
    SettingLightcurveDirectlyWarning,
    UnknownObservatoryWarning,
)
from dk154_targets.obs_info import ObservatoryInfo
from dk154_targets.plotters.visibility_plotter import (
    VisibilityPlotter,
    plot_visibility,
)
from dk154_targets.target import Target, TargetData


@pytest.fixture
def test_target():
    return Target("T101", ra=45.0, dec=60.0)


@pytest.fixture
def test_observer():
    location = EarthLocation(lat=55.6802, lon=12.5724, height=0.0)
    return Observer(location, name="ucph")


class Test__VisibilityPlotter:
    def test__init(self, test_observer):
        t_ref = Time(60000.0, format="mjd")

        plotter = VisibilityPlotter(test_observer, t_ref=t_ref)

        assert np.isclose(plotter.t_grid[0].mjd, 60000.0)
        assert np.isclose(plotter.t_grid[-1].mjd, 60001.0)

        assert isinstance(plotter.obs_info, ObservatoryInfo)
        assert isinstance(plotter.obs_info.moon_altaz, SkyCoord)  # apparenlty not AltAz
        assert isinstance(plotter.obs_info.sun_altaz, SkyCoord)
        assert hasattr(plotter.obs_info.moon_altaz, "alt")
        assert hasattr(plotter.obs_info.sun_altaz, "alt")
        assert plotter.obs_info.target_altaz is None

        assert plotter.altitude_plotted is False
        assert plotter.sky_plotted is False
        assert plotter.moon_plotted is False
        assert plotter.sun_plotted is False
        assert plotter.axes_formatted is False

    def test__plot_class_method(self, test_target, test_observer, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_class_method.pdf"
        assert not fig_path.exists()

        plotter = VisibilityPlotter.plot(test_observer, test_target, t_ref=t_ref)

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 8.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.altitude_plotted
        assert plotter.sky_plotted
        assert plotter.moon_plotted
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_alt(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_alt.pdf"
        assert not fig_path.exists()

        dt = 1.0 / 24.0
        plotter = VisibilityPlotter.plot(
            test_observer, test_target, t_ref=t_ref, alt_ax=False, dt=dt
        )
        assert len(plotter.t_grid) == 25  # 24, + endpoint.

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 4.0])
        assert plotter.alt_ax is None
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.altitude_plotted is False
        assert plotter.sky_plotted
        assert plotter.moon_plotted
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_sky(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_sky.pdf"
        assert not fig_path.exists()

        dt = 1.0 / 24.0
        plotter = VisibilityPlotter.plot(
            test_observer, test_target, t_ref=t_ref, sky_ax=False, dt=dt
        )
        assert len(plotter.t_grid) == 25  # 24, + endpoint.

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 4.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert plotter.sky_ax is None

        assert plotter.altitude_plotted
        assert plotter.sky_plotted is False
        assert plotter.moon_plotted
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_sun(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_sun.pdf"
        assert not fig_path.exists()

        dt = 1.0 / 24.0
        plotter = VisibilityPlotter.plot(
            test_observer, test_target, t_ref=t_ref, sun=False, dt=dt
        )
        assert len(plotter.t_grid) == 25  # 24, + endpoint.

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 8.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.altitude_plotted
        assert plotter.sky_plotted
        assert plotter.moon_plotted
        assert plotter.sun_plotted is False
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_moon(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_moon.pdf"
        assert not fig_path.exists()

        dt = 1.0 / 24.0
        plotter = VisibilityPlotter.plot(
            test_observer, test_target, t_ref=t_ref, moon=False, dt=dt
        )
        assert len(plotter.t_grid) == 25  # 24, + endpoint.

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 8.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.altitude_plotted
        assert plotter.sky_plotted
        assert plotter.moon_plotted is False
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()
