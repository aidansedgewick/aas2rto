import pytest

import numpy as np

import matplotlib.pyplot as plt

import sncosmo

from aas2rto.target import Target
from aas2rto.plotting.salt_param_plotter import (
    SaltParamPlottingWrapper,
    SaltParamPlotter,
    load_c_datasets,
    load_x1_datasets,
)


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
def plotting_util():
    return SaltParamPlotter()


class Test__InitParamPlottingUtil:
    def test__init_util(self):
        # Act
        pl_util = SaltParamPlotter()

        # Assert
        assert not pl_util.literature_plotted
        assert not pl_util.target_plotted
        assert not pl_util.axes_formatted

        # Cleanup
        plt.close(pl_util.fig)


class Test__UtilLitData:
    def test__plot_lit_data(self, plotting_util: SaltParamPlotter):
        # Act
        plotting_util.plot_literature()

        # Assert
        assert plotting_util.literature_plotted
        assert len(plotting_util.lit_handles) == 4

        # Cleanup
        plt.close(plotting_util.fig)


class Test__UtilPlotTarget:
    def test__util_plot_target(
        self,
        basic_target: Target,
        salt_model: sncosmo.Model,
        plotting_util: SaltParamPlotter,
    ):
        # Arrage
        basic_target.models["sncosmo_salt"] = salt_model

        # Act
        plotting_util.plot_target(basic_target)

        # Assert
        assert plotting_util.target_plotted
        assert not plotting_util.intervals_plotted

        # Cleanup
        plt.close(plotting_util.fig)

    def test__util_plot_target_samples(
        self,
        basic_target: Target,
        salt_model_with_samples: sncosmo.Model,
        plotting_util: SaltParamPlotter,
    ):
        # Arrage
        basic_target.models["sncosmo_salt"] = salt_model_with_samples

        # Act
        plotting_util.plot_target(basic_target)

        # Assert
        assert plotting_util.target_plotted
        assert plotting_util.intervals_plotted

        # Cleanup
        plt.close(plotting_util.fig)

    def test__no_model_no_fail(
        self, basic_target: Target, plotting_util: SaltParamPlotter
    ):
        # Act
        plotting_util.plot_target(basic_target)

        # Assert
        assert not plotting_util.target_plotted

        # Cleanup
        plt.close(plotting_util.fig)


class Test__UtilPlotClsMethod:
    def test__plot_method(self, basic_target: Target, salt_model: sncosmo.Model):
        # Arrange
        basic_target.models["sncosmo_salt"] = salt_model

        # Act
        pl_util = SaltParamPlotter.plot(basic_target)

        # Assert
        assert isinstance(pl_util.fig, plt.Figure)
        assert pl_util.target_plotted
        assert pl_util.literature_plotted
        assert pl_util.axes_formatted

        # Cleanup
        plt.close(pl_util.fig)


class Test__PlotterInit:
    def test__param_plotter_init(self):
        # Act
        plot_func = SaltParamPlottingWrapper()

        # Assert
        assert isinstance(plot_func.x1_data, dict)
        assert isinstance(plot_func.c_data, dict)


class Test__PlotterCall:
    def test__plotter_call(self, basic_target: Target, salt_model: sncosmo.Model):
        # Arrange
        plot_func = SaltParamPlottingWrapper()
        basic_target.models["sncosmo_salt"] = salt_model

        # Act
        fig = plot_func(basic_target)

        # Cleanup
        plt.close(fig)

    def test__call_with_samples(self, basic_target: Target, salt_model: sncosmo.Model):
        # Arrange
        plot_func = SaltParamPlottingWrapper()
        basic_target.models["sncosmo_salt"] = salt_model

        # Act
        fig = plot_func(basic_target)

        # Cleanup
        plt.close(fig)

    def test__call_no_model(
        self, basic_target: Target, salt_model_with_samples: sncosmo.Model
    ):
        # Arrange
        plot_func = SaltParamPlottingWrapper()
        basic_target.models["sncosmo_salt"] = salt_model_with_samples

        # Act
        fig = plot_func(basic_target)

        # Cleanup
        plt.close(fig)
