import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

import sncosmo

from dk154_targets.target import Target, TargetData
from dk154_targets.modeling.sncosmo import SncosmoSaltModeler, initialise_model
from dk154_targets.plotters.sncosmo_plotter import SncosmoLightcurvePlotter


@pytest.fixture
def fixed_t0():
    t0 = 60020.0
    return t0


# @pytest.fixture
# def t0_jd(fixed_t0):
#     return fixed_t0 + 2_400_000.5


@pytest.fixture
def fixed_params(fixed_t0):
    return dict(z=0.05, t0=fixed_t0, x0=2e-3, x1=-0.5, c=0.0)


@pytest.fixture  # (scope="module")
def true_model(fixed_params):
    model = initialise_model()
    model.update(fixed_params)
    return model


@pytest.fixture
def samples(fixed_params):
    N = 500
    samples = np.column_stack(
        [
            fixed_params["z"] + np.random.normal(0, 0.005, N),
            fixed_params["t0"] + np.random.normal(0, 1.0, N),
            fixed_params["x0"] + np.random.normal(0, 4e-4, N),
            fixed_params["x1"] + np.random.normal(0, 0.1, N),
            fixed_params["c"] + np.random.normal(0, 5e-2, N),
        ]
    )
    return samples


@pytest.fixture
def model_with_samples(true_model, samples):
    result = {"samples": samples, "vparam_names": "z t0 x0 x1 c".split()}
    true_model.result = result
    return true_model


@pytest.fixture
def dt_vals():
    before = [-16.0, -15.0, -12.5, -11.0, -8.0, -5.5, -4.0, -2.0, -0.5]
    after = [0.5, 3.0, 5.0, 7.5, 11.0, 14.0, 18.0, 22.0]
    return before + after


@pytest.fixture
def mock_lc(true_model, fixed_t0, dt_vals):
    """
    photometry samples are drawn FROM THE MODEL!
    This is so we're not testing the actual fitting here. just plotting.
    """

    dt = np.array(dt_vals)
    ztfg_t_grid = fixed_t0 + dt
    ztfr_t_grid = fixed_t0 + dt + 0.1
    gmag = true_model.bandmag("ztfg", "ab", ztfg_t_grid)
    gmag = np.random.normal(gmag, 0.2, len(ztfg_t_grid))
    rmag = true_model.bandmag("ztfr", "ab", ztfr_t_grid)
    rmag = np.random.normal(rmag, 0.2, len(ztfr_t_grid))
    gmag_tag = ["valid"] * len(dt)
    rmag_tag = ["valid"] * len(dt)
    gmag_tag[0] = "badqual"
    gmag_tag[2] = "badqual"
    gmag_tag[-1] = "badqual"
    rmag_tag[1] = "badqual"
    band_col = np.array(["ztfg"] * len(dt) + ["ztfr"] * len(dt))

    t_grid = np.concatenate([ztfg_t_grid, ztfr_t_grid])
    mag = np.concatenate([gmag, rmag])
    magerr = np.array([0.1] * len(t_grid))
    tagcol = np.concatenate([gmag_tag, rmag_tag])

    df = pd.DataFrame(
        {"mjd": t_grid, "mag": mag, "magerr": magerr, "band": band_col, "tag": tagcol}
    )
    df.sort_values("mjd", inplace=True, ignore_index=True)
    return df


@pytest.fixture
def mock_target(mock_lc):
    t = Target("T101", ra=30.0, dec=60.0)
    t.compiled_lightcurve = mock_lc

    t.score_comments["no_observatory"] = ["some comment here", "another comment"]
    return t


@pytest.fixture
def mock_target_model(mock_target, true_model):
    mock_target.models["sncosmo_salt"] = true_model
    return mock_target


@pytest.fixture
def mock_target_model_samples(mock_target, model_with_samples):
    mock_target.models["sncosmo_salt"] = model_with_samples
    return mock_target


class Test__PlotterInit:

    def test__plotter_init(self):
        t_ref = Time(60045, format="mjd")

        plotter = SncosmoLightcurvePlotter(t_ref=t_ref, grid_dt=1.0)

        assert np.isclose(plotter.grid_dt, 1.0)
        assert np.isclose(plotter.forecast_days, 15.0)

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False
        assert plotter.models_plotted is False
        assert plotter.samples_plotted is False


class Test__Plotter:

    def test__plotter_normal(self, mock_target_model_samples, tmp_path):
        t_ref = Time(60045, format="mjd")

        plotter = SncosmoLightcurvePlotter.plot(
            mock_target_model_samples, t_ref=t_ref, grid_dt=1.0
        )

        assert plotter.photometry_plotted is True
        # assert plotter.cutouts_added is False
        assert plotter.axes_formatted is True
        assert plotter.comments_added is True
        assert plotter.models_plotted is True
        assert plotter.samples_plotted is True

        fig_path = tmp_path / "target_with_samples.png"
        plotter.fig.savefig(fig_path)

    def test__plotter_no_samples(self, mock_target_model, tmp_path):
        t_ref = Time(60045.0, format="mjd")

        plotter = SncosmoLightcurvePlotter(t_ref=t_ref, grid_dt=1.0)

        plotter.plot_sncosmo_models(mock_target_model)

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False
        assert plotter.models_plotted is True
        assert plotter.samples_plotted is False

        fig_path = tmp_path / "target_model_no_samples_only.png"
        plotter.fig.savefig(fig_path)

    def test__plotter_with_samples(self, mock_target_model_samples, tmp_path):
        t_ref = Time(60045, format="mjd")

        plotter = SncosmoLightcurvePlotter(t_ref=t_ref, grid_dt=1.0)

        plotter.plot_sncosmo_models(mock_target_model_samples)

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False
        assert plotter.models_plotted is True
        assert plotter.samples_plotted is True

        fig_path = tmp_path / "target_model_with_samples_only.png"
        plotter.fig.savefig(fig_path)
