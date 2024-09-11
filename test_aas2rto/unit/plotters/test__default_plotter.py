import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from astroplan import FixedTarget, Observer

from aas2rto import Target, TargetData
from aas2rto.exc import MissingDateError, UnknownObservatoryWarning
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.plotters.default_plotter import (
    DefaultLightcurvePlotter,
    plot_default_lightcurve,
)
from aas2rto.target import SettingLightcurveDirectlyWarning


@pytest.fixture
def test_lc_g_rows():
    return [
        (60000.0, 1, 20.5, 0.1, 20.2, "upperlim"),
        (60001.0, 2, 20.4, 0.1, 20.3, "upperlim"),
        (60002.0, 3, 20.3, 0.1, 20.2, "upperlim"),
        (60003.0, 4, 20.2, 0.1, 20.1, "upperlim"),
        (60004.0, 5, 20.1, 0.2, 20.0, "badqual"),
        (60005.0, 6, 20.1, 0.2, 20.0, "badqual"),
        (60006.0, 7, 20.1, 0.2, 20.0, "badqual"),
        (60007.0, 8, 20.0, 0.1, 20.0, "valid"),
        (60008.0, 9, 19.7, 0.1, 20.0, "valid"),
        (60009.0, 10, 19.5, 0.1, 20.0, "valid"),
    ]


@pytest.fixture
def test_lc_r_rows():
    return [
        (60000.5, 1, 19.5, 0.1, 19.2, "upperlim"),
        (60001.5, 2, 19.4, 0.1, 19.3, "upperlim"),
        (60002.5, 3, 19.3, 0.1, 19.2, "upperlim"),
        (60003.5, 4, 19.2, 0.1, 19.1, "upperlim"),
        (60004.5, 5, 19.1, 0.2, 19.0, "badqual"),
        (60005.5, 6, 19.1, 0.2, 19.0, "badqual"),
        (60006.5, 7, 19.1, 0.2, 19.0, "badqual"),
        (60007.5, 8, 19.0, 0.1, 19.0, "valid"),
        (60008.5, 9, 18.7, 0.1, 19.0, "valid"),
        (60009.5, 10, 19.4, 0.1, 19.0, "valid"),
    ]


@pytest.fixture
def mock_lc(test_lc_g_rows, test_lc_r_rows):
    columns = "mjd obsId mag magerr diffmaglim tag".split()
    g_df = pd.DataFrame(test_lc_g_rows, columns=columns)
    g_df.loc[:, "band"] = "ztfg"
    r_df = pd.DataFrame(test_lc_r_rows, columns=columns)
    r_df.loc[:, "band"] = "ztfr"

    lc = pd.concat([g_df, r_df], ignore_index=True)
    lc.loc[:, "obsId"] = np.arange(1, len(lc) + 1)
    lc.sort_values("mjd", inplace=True)
    return lc


@pytest.fixture
def mock_cutouts():
    # These are more elaborate than necessary...
    grid = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(grid, grid)
    template = xx**2 + yy**2
    source = np.maximum(20 - (5 * xx**2 + 5 * yy**2), 0.0)
    return {
        "science": template + source,
        "difference": source,
        "template": template,
    }


def basic_ztf_lc_compiler(target):
    lc = target.target_data["fink"].lightcurve.copy()
    lc.loc[:, "jd"] = Time(lc["mjd"], format="mjd").jd
    return lc


@pytest.fixture
def mock_target(mock_lc):
    t = Target("T101", ra=45.0, dec=60.0)
    fink_data = t.get_target_data("fink")
    fink_data.add_lightcurve(mock_lc)
    t.compiled_lightcurve = basic_ztf_lc_compiler(t)
    return t


@pytest.fixture
def test_observer():
    location = EarthLocation(lat=55.6802, lon=12.5724, height=0.0)
    return Observer(location, name="ucph")


class Test__DefaultLightcurvePlotter:
    def test__init(self):
        plotter = DefaultLightcurvePlotter()

        assert hasattr(plotter, "lc_gs")
        assert hasattr(plotter, "zscaler")

        assert hasattr(plotter, "default_figsize")

        assert hasattr(plotter, "ztf_colors")
        assert hasattr(plotter, "atlas_colors")
        assert hasattr(plotter, "det_kwargs")
        assert hasattr(plotter, "ulimit_kwargs")
        assert hasattr(plotter, "badqual_kwargs")

        assert hasattr(plotter, "tag_col")
        assert hasattr(plotter, "valid_tag")
        assert hasattr(plotter, "badqual_tag")
        assert hasattr(plotter, "ulimit_tag")

        assert hasattr(plotter, "band_col")

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False

    def test__fig_init(self):
        plotter = DefaultLightcurvePlotter()
        assert isinstance(plotter.fig, plt.Figure)
        assert isinstance(plotter.ax, plt.Axes)

    def test__no_exception_plot_photometry_blank(self, mock_target, tmp_path):
        t_ref = Time(60012.0, format="mjd")
        fig_path = tmp_path / "lc_no_photom.pdf"
        assert not fig_path.exists()

        mock_target.target_data["fink"] = None
        mock_target.compiled_lightcurve = None

        plotter = DefaultLightcurvePlotter(t_ref=t_ref)

        plotter.plot_photometry(mock_target)

        assert plotter.photometry_plotted is False

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_photometry(self, mock_target, tmp_path):
        t_ref = Time(60015.0, format="mjd")
        fig_path = tmp_path / "lc_with_photom.pdf"
        assert not fig_path.exists()

        plotter = DefaultLightcurvePlotter(t_ref=t_ref)
        plotter.plot_photometry(mock_target)

        assert plotter.photometry_plotted is True
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__add_postage_stamps(self, mock_target, mock_cutouts, tmp_path):
        mock_target.target_data["fink"].cutouts = mock_cutouts
        fig_path = tmp_path / "lc_add_stamps.pdf"
        assert not fig_path.exists()

        plotter = DefaultLightcurvePlotter()
        plotter.init_fig()
        plotter.add_cutouts(mock_target)

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is True
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_class_method(self, mock_target, mock_cutouts, tmp_path):
        t_ref = Time(60015.0, format="mjd")
        mock_target.target_data["fink"].cutouts = mock_cutouts
        mock_target.target_data["tns"] = TargetData(parameters={"Redshift": 0.1})
        mock_target.score_comments["no_observatory"] = [
            "comm a",
            "comm b",
            "comm c",
            "comm d",
            "comm e",
            "comm f",
        ]
        plotter = DefaultLightcurvePlotter.plot(mock_target, t_ref=t_ref)
        fig_path = tmp_path / "lc_plotter_class_method.pdf"
        assert not fig_path.exists()

        assert plotter.photometry_plotted
        assert plotter.cutouts_added
        assert plotter.axes_formatted
        assert plotter.comments_added

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()
