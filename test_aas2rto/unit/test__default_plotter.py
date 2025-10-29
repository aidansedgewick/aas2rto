import pytest
from typing import Callable, Dict

import numpy as np

import matplotlib.pyplot as plt

from astropy.time import Time

from aas2rto.exc import (
    MissingColumnWarning,
    UnexpectedKeysWarning,
    UnknownPhotometryTagWarning,
)
from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler
from aas2rto.plotting.default_plotter import (
    DefaultLightcurvePlotter,
    plot_default_lightcurve,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def lc_plotter(t_plot: Time):
    return DefaultLightcurvePlotter(t_ref=t_plot)


class Test__PlotterInit:
    def test__plotter_init(self, t_plot: Time):
        # Act
        plotter = DefaultLightcurvePlotter(t_ref=t_plot)

        # Assert
        assert isinstance(plotter.fig, plt.Figure)
        assert isinstance(plotter.ax, plt.Axes)

        assert isinstance(plotter.valid_kwargs, dict)
        assert isinstance(plotter.ulimit_kwargs, dict)
        assert isinstance(plotter.badqual_kwargs, dict)

        assert isinstance(plotter.tag_col, str)
        assert isinstance(plotter.mag_col, str)
        assert isinstance(plotter.magerr_col, str)
        assert isinstance(plotter.diffmaglim_col, str)

        assert isinstance(plotter.valid_tag, str)
        assert isinstance(plotter.ulimit_tag, str)
        assert isinstance(plotter.badqual_tag, str)

        assert not plotter.photometry_plotted
        assert not plotter.cutouts_added
        assert not plotter.axes_formatted
        assert not plotter.comments_added


class Test__PlotPhotomMethod:
    def test__plot_photom(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Act
        lc_plotter.plot_photometry(target_to_plot)

        # Assert
        assert lc_plotter.photometry_plotted

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_lc_no_fail(
        self, lc_plotter: DefaultLightcurvePlotter, basic_target: Target
    ):
        # Act
        with pytest.warns(UserWarning):
            lc_plotter.plot_photometry(basic_target)

        # Assert
        assert not lc_plotter.photometry_plotted

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_band_col(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Arrange
        target_to_plot.compiled_lightcurve.drop("band", axis=1, inplace=True)

        # Act
        with pytest.warns(MissingColumnWarning):
            lc_plotter.plot_photometry(target_to_plot)

        # Assert
        assert lc_plotter.photometry_plotted

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_tag_no_fail(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Arrange
        target_to_plot.compiled_lightcurve.drop("tag", axis=1, inplace=True)

        # Act
        with pytest.warns(MissingColumnWarning):
            lc_plotter.plot_photometry(target_to_plot)

        # Assert
        assert lc_plotter.photometry_plotted

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__bad_tag_no_fail(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Arrange
        target_to_plot.compiled_lightcurve.loc[0, "tag"] = "other_tag"

        # Act
        with pytest.warns(UnknownPhotometryTagWarning):
            lc_plotter.plot_photometry(target_to_plot)

        # Assert
        lc_plotter.photometry_plotted = True

        # Cleanup
        plt.close(lc_plotter.fig)


class Test__CutoutsMethod:
    def test__add_cutouts(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Act
        lc_plotter.add_cutouts(target_to_plot)

        # Assert
        assert lc_plotter.cutouts_added

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_cutouts_no_fail(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Arrange
        target_to_plot.target_data["ztf"].cutouts = {}

        # Act
        lc_plotter.add_cutouts(target_to_plot)

        # Assert
        assert not lc_plotter.cutouts_added

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__cutouts_bad_keys_no_fail(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Arrange
        target_to_plot.target_data["ztf"].cutouts = {"blah": np.zeros((10, 10))}

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            lc_plotter.add_cutouts(target_to_plot)

        assert not lc_plotter.cutouts_added

        # Cleanup
        plt.close(lc_plotter.fig)


class Test__CommentsMethod:
    def test__add_comms(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Act
        lc_plotter.add_comments(target_to_plot)

        # Assert
        assert not lc_plotter.photometry_plotted
        assert lc_plotter.comments_added

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_comms_no_fail(
        self, lc_plotter: DefaultLightcurvePlotter, target_to_plot: Target
    ):
        # Arrange
        target_to_plot.science_comments = []

        # Act
        lc_plotter.add_comments(target_to_plot)

        # Assert
        assert not lc_plotter.photometry_plotted
        assert not lc_plotter.comments_added

        # Cleanup
        plt.close(lc_plotter.fig)


class Test__PlotMethod:
    def test__plot_method(self, target_to_plot: Target, t_plot: Time):

        # Act
        plotter = DefaultLightcurvePlotter.plot(target_to_plot, t_ref=t_plot)

        # Assert
        assert plotter.photometry_plotted
        assert plotter.cutouts_added
        assert plotter.axes_formatted
        assert plotter.comments_added

        # Cleanup
        plt.close(plotter.fig)


class Test__PlotterFunc:
    def test__plot_func_return_plotter(self, target_to_plot: Target, t_plot: Time):

        # Act
        plotter = plot_default_lightcurve(
            target_to_plot, t_ref=t_plot, return_plotter=True
        )

        # Assert
        assert plotter.photometry_plotted
        assert plotter.cutouts_added
        assert plotter.axes_formatted
        assert plotter.comments_added

    def test__plot_func(self, target_to_plot: Target, t_plot: Time):
        # Act
        result = plot_default_lightcurve(target_to_plot, t_ref=t_plot)

        # Assert
        assert isinstance(result, plt.Figure)

        # Cleanup
        plt.close(result)
