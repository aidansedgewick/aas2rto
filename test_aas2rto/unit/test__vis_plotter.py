import copy
import pytest

from astroplan import Observer

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt

from aas2rto.ephem_info import EphemInfo
from aas2rto.plotting.visibility_plotter import VisibilityPlotter, plot_visibility
from aas2rto.target import Target


@pytest.fixture
def ephem_target(basic_target: Target, lasilla_ephem: EphemInfo):
    target_ephem = copy.copy(lasilla_ephem)
    target_ephem.set_target_altaz(basic_target.coord)
    basic_target.ephem_info["lasilla"] = target_ephem
    return basic_target


@pytest.fixture
def lasilla_ephem(lasilla: Observer, t_fixed: Time):
    return EphemInfo(lasilla, t_ref=t_fixed, dt=1.5 * u.hour)


@pytest.fixture
def vis_plotter(lasilla: Observer, lasilla_ephem: EphemInfo):
    return VisibilityPlotter(lasilla, ephem_info=lasilla_ephem)


class Test__VisPlotterInit:
    def test__default_init(self, lasilla: Observer):
        # Act
        vp = VisibilityPlotter(lasilla)

        # Assert
        assert isinstance(vp.observatory, Observer)
        assert isinstance(vp.ephem_info, EphemInfo)
        assert isinstance(vp.t_grid, Time)

        assert isinstance(vp.fig, plt.Figure)
        assert isinstance(vp.alt_ax, plt.Axes)
        assert isinstance(vp.sky_ax, plt.Axes)

        assert not vp.altitude_plotted
        assert not vp.sky_plotted
        assert not vp.moon_plotted
        assert not vp.sun_plotted
        assert not vp.axes_formatted

        # Cleanup
        plt.close(vp.fig)

    def test__init_use_existing_ephem(
        self, lasilla: Observer, lasilla_ephem: EphemInfo
    ):
        # Act
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem)

        # Assert
        assert id(vp.ephem_info) == id(lasilla_ephem)

        # Cleanup
        plt.close(vp.fig)

    def test__init_no_sky(self, lasilla: Observer, lasilla_ephem: EphemInfo):
        # Act
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem, sky_ax=False)
        assert isinstance(vp.fig, plt.Figure)
        assert isinstance(vp.alt_ax, plt.Axes)
        assert vp.sky_ax is None

        # Cleanup
        plt.close(vp.fig)

    def test__init_no_alt(self, lasilla: Observer, lasilla_ephem: EphemInfo):
        # Act
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem, alt_ax=False)
        assert isinstance(vp.fig, plt.Figure)
        assert vp.alt_ax is None
        assert isinstance(vp.sky_ax, plt.Axes)

        # Cleanup
        plt.close(vp.fig)


class Test__PlotTargetMethod:
    def test__plot_target(
        self, lasilla: Observer, lasilla_ephem: EphemInfo, ephem_target: Target
    ):
        # Arrange
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem)

        # Act
        vp.plot_target(ephem_target)

        # Assert
        assert vp.altitude_plotted
        assert vp.sky_plotted
        assert not vp.moon_plotted
        assert not vp.sun_plotted
        assert not vp.axes_formatted

        # Cleanup
        plt.close(vp.fig)

    def test__no_sky(
        self, lasilla: Observer, lasilla_ephem: EphemInfo, ephem_target: Target
    ):
        # Arrange
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem, sky_ax=False)

        # Act
        vp.plot_target(ephem_target)

        # Assert
        assert vp.altitude_plotted
        assert not vp.sky_plotted
        assert not vp.moon_plotted
        assert not vp.sun_plotted
        assert not vp.axes_formatted

        # Cleanup
        plt.close(vp.fig)

    def test__plot_target_no_alt(
        self, lasilla: Observer, lasilla_ephem: EphemInfo, ephem_target: Target
    ):
        # Arrange
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem, alt_ax=False)

        # Act
        vp.plot_target(ephem_target)

        # Assert
        assert not vp.altitude_plotted
        assert vp.sky_plotted
        assert not vp.moon_plotted
        assert not vp.sun_plotted
        assert not vp.axes_formatted

        # Cleanup
        plt.close(vp.fig)

    def test__target_no_ephem(
        self, lasilla: Observer, lasilla_ephem: EphemInfo, basic_target: Target
    ):
        # Arrange
        vp = VisibilityPlotter(lasilla, ephem_info=lasilla_ephem)
        assert set(basic_target.ephem_info.keys()) == set([])

        # Act
        vp.plot_target(basic_target)

        # Assert
        assert vp.altitude_plotted
        assert vp.sky_plotted

        # Cleanup
        plt.close(vp.fig)

    def test__plot_target_sky_coord(self, vis_plotter: VisibilityPlotter):
        # Arrange
        sky_coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Act
        vis_plotter.plot_target(sky_coord)

        # Assert
        assert vis_plotter.altitude_plotted
        assert vis_plotter.sky_plotted

        # Cleanup
        plt.close(vis_plotter.fig)

    def test__plot_fails_bad_coord(self, vis_plotter: VisibilityPlotter):
        # Act
        with pytest.raises(TypeError):
            vis_plotter.plot_target(None)

        # Cleanup
        plt.close(vis_plotter.fig)


class Test__PlotClassMethod:
    def test__all_options(
        self, lasilla: Observer, lasilla_ephem: EphemInfo, ephem_target: Target
    ):
        # Act
        vp = VisibilityPlotter.plot(lasilla, ephem_target, ephem_info=lasilla_ephem)

        # Assert
        assert isinstance(vp, VisibilityPlotter)

        assert vp.altitude_plotted
        assert vp.sky_plotted
        assert vp.moon_plotted
        assert vp.sun_plotted
        assert vp.axes_formatted

        # Cleanup
        plt.close(vp.fig)
