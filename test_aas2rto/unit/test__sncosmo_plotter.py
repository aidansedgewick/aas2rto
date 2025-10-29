import pytest
from typing import Dict

import numpy as np

import matplotlib.pyplot as plt

from astropy.time import Time

import sncosmo

from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler
from aas2rto.plotting.sncosmo_plotter import (
    SncosmoLightcurvePlotter,
    plot_sncosmo_lightcurve,
    get_sample_quartiles,
    get_model_median_params,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def t_plot():
    return Time(60010.0, format="mjd")


@pytest.fixture
def salt_model():
    model = sncosmo.Model(source="salt3")
    model.update(dict(z=0.02, x0=1e-2, x1=0.0, c=0.0, t0=60010.0))
    result = dict(chisq=5.0, ndof=5, vparam_names=["z", "x0", "x1", "c", "t0"])
    model.result = result
    return model


@pytest.fixture
def salt_model_with_samples(salt_model):
    mu_vals = [salt_model[p] for p in "z x0 x1 c t0".split()]
    sig_vals = [0.005, 1e-4, 0.05, 0.01, 0.3]

    samples = np.column_stack(
        [np.random.normal(mu, sig, 500) for mu, sig in zip(mu_vals, sig_vals)]
    )

    salt_model.result["samples"] = samples
    return salt_model


@pytest.fixture
def target_with_models(target_to_plot: Target, salt_model: sncosmo.Model):
    target_to_plot.models["sncosmo_salt"] = salt_model
    return target_to_plot


@pytest.fixture
def target_with_samples(target_to_plot: Target, salt_model_with_samples: sncosmo.Model):
    target_to_plot.models["sncosmo_salt"] = salt_model_with_samples
    return target_to_plot


@pytest.fixture
def lc_plotter(t_plot: Time):
    return SncosmoLightcurvePlotter(t_ref=t_plot)


class Test__SampleQuartilesHelper:
    def test__get_samples(self, salt_model_with_samples: sncosmo.Model):
        # Arrange
        t_grid = np.arange(60000.0, 60020.0, 1.0)

        # Act
        lc_bounds = get_sample_quartiles(
            t_grid, salt_model_with_samples, "ztfg", spacing=1, q=[0.16, 0.84]
        )

        # Assert
        assert lc_bounds.ndim == 2

        assert all(lc_bounds[0] < lc_bounds[1])  # upper bound brighter mag


class Test__MedianModelHelper:
    def test__get_model_median(self, salt_model_with_samples: sncosmo.Model):
        # Arrange
        bad_pdict = dict(z=np.nan, x0=np.nan, x1=np.nan, c=np.nan, t0=np.nan)
        salt_model_with_samples.update(bad_pdict)

        # Act
        med_model = get_model_median_params(salt_model_with_samples)

        # Assert
        assert id(salt_model_with_samples) != id(med_model)  # ie. it's a copy

        assert np.isclose(med_model["z"], 0.02, rtol=0.1)
        assert np.isclose(med_model["x0"], 1e-2, rtol=0.1)
        assert np.isclose(med_model["x1"], 0.0, atol=0.01)
        assert np.isclose(med_model["c"], 0.0, atol=5e-3)
        assert np.isclose(med_model["t0"], 60010.0, atol=1.0)


class Test__PlotterInit:
    def test__plotter_init(self):
        # Act
        plotter = SncosmoLightcurvePlotter()

        # Assert
        assert not plotter.photometry_plotted  # inherited
        assert not plotter.cutouts_added  # inherited
        assert not plotter.axes_formatted  # inherited
        assert not plotter.comments_added  # inherited
        assert not plotter.target_has_models
        assert not plotter.models_plotted
        assert not plotter.samples_plotted

        # Cleanup
        plt.close(plotter.fig)


class Test__PlotModel:
    def test__plot_model(
        self, target_with_models: Target, lc_plotter: SncosmoLightcurvePlotter
    ):
        # Act
        lc_plotter.plot_sncosmo_models(target_with_models)

        # Assert
        assert not lc_plotter.photometry_plotted
        assert lc_plotter.models_plotted
        assert not lc_plotter.samples_plotted

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_model_no_fail(
        self, target_to_plot: Target, lc_plotter: SncosmoLightcurvePlotter
    ):
        # Arrange
        assert set(target_to_plot.models) == set()

        # Act
        lc_plotter.plot_sncosmo_models(target_to_plot)

        # Assert
        assert not lc_plotter.photometry_plotted
        assert not lc_plotter.models_plotted
        assert not lc_plotter.samples_plotted

        # Cleanup
        plt.close(lc_plotter.fig)

    def test__no_lc_no_fail(
        self, target_with_models: Target, lc_plotter: SncosmoLightcurvePlotter
    ):
        # Arrange
        target_with_models.compiled_lightcurve = None

        # Assert
        lc_plotter.plot_sncosmo_models(target_with_models)

        # Assert
        assert not lc_plotter.models_plotted

        # Cleanup
        plt.close(lc_plotter.fig)


class Test__PlotWithSamples:
    def test__plot_with_samples(
        self, target_with_samples: Target, lc_plotter: SncosmoLightcurvePlotter
    ):
        # Act
        lc_plotter.plot_sncosmo_models(target_with_samples)

        # Assert
        assert lc_plotter.models_plotted
        assert lc_plotter.samples_plotted

        # Cleanup
        plt.close(lc_plotter.fig)


class Test__PlotMethod:
    def test__plot_method(self, target_with_samples: Target, t_plot: Time):
        # Act
        plotter = SncosmoLightcurvePlotter.plot(target_with_samples, t_ref=t_plot)

        # Assert
        assert plotter.photometry_plotted
        assert plotter.cutouts_added
        assert plotter.axes_formatted
        assert plotter.comments_added
        assert plotter.models_plotted
        assert plotter.samples_plotted

        # Cleanup
        plt.close(plotter.fig)


class Test__PlotFunc:

    def test__return_plotter(self, target_with_samples: Target, t_plot: Time):
        # Act
        plotter = plot_sncosmo_lightcurve(
            target_with_samples, t_ref=t_plot, return_plotter=True
        )

        # Assert
        assert isinstance(plotter, SncosmoLightcurvePlotter)
        assert plotter.photometry_plotted
        assert plotter.cutouts_added
        assert plotter.axes_formatted
        assert plotter.comments_added
        assert plotter.models_plotted
        assert plotter.samples_plotted

        # Cleanup
        plt.close(plotter.fig)

    def test__figure(self, target_with_samples: Target, t_plot: Time):
        # Act
        result = plot_sncosmo_lightcurve(target_with_samples, t_ref=t_plot)

        # Assert
        assert isinstance(result, plt.Figure)

        # Cleanup
        plt.close(result)
