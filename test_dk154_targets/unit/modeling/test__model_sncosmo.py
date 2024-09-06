import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.table import Table

import sncosmo

from dk154_targets.modeling.sncosmo import (
    get_detections,
    build_astropy_lightcurve,
    initialise_model,
    write_salt_model,
    read_salt_model,
    SncosmoSaltModeler,
)
from dk154_targets.target import Target, TargetData
from dk154_targets.exc import UnexpectedKeysWarning


@pytest.fixture
def fixed_t0():
    t0 = 60020.0
    return t0


@pytest.fixture
def fixed_params(fixed_t0):
    return dict(z=0.05, t0=fixed_t0, x0=2e-3, x1=-1.0, c=0.0)


@pytest.fixture  # (scope="module")
def true_model(fixed_params):
    model = initialise_model()
    model.update(fixed_params)
    return model


@pytest.fixture
def dt_vals():
    before = [-16.0, -15.0, -12.5, -11.0, -8.0, -5.5, -4.0, -2.0, -0.5]
    after = [0.5, 3.0, 5.0, 7.5, 11.0, 14.0, 18.0, 22.0]
    return before + after


@pytest.fixture
def mock_lc(true_model, fixed_t0, dt_vals):
    dt = np.array(dt_vals)
    ztfg_t_grid = fixed_t0 + dt
    ztfr_t_grid = fixed_t0 + dt + 0.1
    gmag = true_model.bandmag("ztfg", "ab", ztfg_t_grid)
    rmag = true_model.bandmag("ztfr", "ab", ztfr_t_grid)
    gmag_tag = ["valid"] * len(dt)
    rmag_tag = ["valid"] * len(dt)
    gmag_tag[0] = "badquality"
    gmag_tag[2] = "badquality"
    gmag_tag[-1] = "badquality"
    rmag_tag[1] = "badquality"
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
    return t


class Test__GetDetections:
    def test__get_detections(self, mock_target):
        detections = get_detections(
            mock_target.compiled_lightcurve, badqual_tag="badquality"
        )

        assert len(detections) == 34

    def test__get_detections_badqual(self, mock_target):
        detections = get_detections(mock_target.compiled_lightcurve, use_badqual=False)
        assert len(detections) == 30

    def test__get_detections_no_tag(self, mock_target):
        mock_target.compiled_lightcurve.drop("tag", axis=1, inplace=True)
        assert "tag" not in mock_target.compiled_lightcurve.columns

        detections = get_detections(mock_target.compiled_lightcurve)

        assert len(detections) == 34

    def test__get_detections_no_tag_usebadqual_false(self, mock_target):
        mock_target.compiled_lightcurve.drop("tag", axis=1, inplace=True)
        assert "tag" not in mock_target.compiled_lightcurve.columns

        detections = get_detections(mock_target.compiled_lightcurve, use_badqual=False)
        assert len(detections) == 34


class Test__BuildAstropyLc:
    def test__build_astropy_detections(self, mock_lc):
        result = build_astropy_lightcurve(mock_lc)
        assert isinstance(result, Table)

        assert "flux" in result.columns
        assert "fluxerr" in result.columns
        assert "snr" in result.columns
        assert "zp" in result.columns
        assert "zpsys" in result.columns


class Test__InitModel:
    def test__init_model(self):
        model = initialise_model()
        assert isinstance(model, sncosmo.Model)
        assert model.source.name == "salt2"
        assert set(model.effect_names) == set(["mw"])


class Test__InitModeler:
    def test__init(self):
        modeler = SncosmoSaltModeler()

        assert modeler.use_emcee is True
        assert np.isclose(modeler.faint_limit, 99.0)
        assert modeler.use_badqual is True
        assert modeler.existing_models_path is None

    def test__kwargs(self):

        modeler = SncosmoSaltModeler(use_emcee=False)
        assert modeler.use_emcee is False
        assert np.isclose(modeler.faint_limit, 99.0)
        assert modeler.use_badqual is True

        modeler = SncosmoSaltModeler(faint_limit=20.0)
        assert modeler.use_emcee is True
        assert np.isclose(modeler.faint_limit, 20.0)
        assert modeler.use_badqual is True

        modeler = SncosmoSaltModeler(use_badqual=False)
        assert modeler.use_emcee is True
        assert np.isclose(modeler.faint_limit, 99.0)
        assert modeler.use_badqual is False


class Test__Modeling:
    def test__lsq_fit(self, mock_target):
        modeler = SncosmoSaltModeler(use_emcee=False)

        model = modeler(mock_target)

        assert isinstance(model, sncosmo.Model)
        assert 60004.0 < model["t0"]
        assert model["t0"] < 60044.0
        assert hasattr(model, "result")
        assert "samples" not in model.result
        assert len(model.result.vparam_names) == 5
        assert set(model.result.vparam_names) == set("z t0 x0 x1 c".split())

    def test__lsq_fit_fixed_redshift(self, mock_target):
        modeler = SncosmoSaltModeler(use_emcee=False)
        tns_data = mock_target.get_target_data("tns")
        tns_data.parameters = {"Redshift": 0.05}

        model = modeler(mock_target)

        assert isinstance(model, sncosmo.Model)
        assert 60004.0 < model["t0"]
        assert model["t0"] < 60044.0
        assert hasattr(model, "result")
        assert "samples" not in model.result
        assert len(model.result.vparam_names) == 4
        assert set(model.result.vparam_names) == set("t0 x0 x1 c".split())

    def test__no_exc_on_lsq_error(self, mock_target, monkeypatch):
        modeler = SncosmoSaltModeler()

        def mock_fit_lc(args, **kwargs):
            raise ValueError

        with monkeypatch.context() as m:
            m.setattr("sncosmo.fit_lc", mock_fit_lc)

            model = modeler(mock_target)

        assert model is None

    def test__with_emcee(self, mock_target):
        modeler = SncosmoSaltModeler(nsamples=200, nwalkers=10)

        model = modeler(mock_target)

        assert isinstance(model, sncosmo.Model)
        assert 60004.0 < model["t0"]
        assert model["t0"] < 60044.0
        assert hasattr(model, "result")
        assert "samples" in model.result
        samples = model.result["samples"]
        assert samples.shape == (2000, 5)  # nsamples * nwalkers

    def test__no_exc_on_failing_emcee(self, mock_target, monkeypatch):
        modeler = SncosmoSaltModeler()

        def mock_mcmc_lc(*args, **kwargs):
            raise ValueError

        with monkeypatch.context() as m:
            m.setattr("sncosmo.mcmc_lc", mock_mcmc_lc)

            model = modeler(mock_target)

        assert modeler.use_emcee is True
        assert "samples" not in model.result


class Test__ModelReadWrite:
    def test__models_save(self, true_model, tmp_path):

        model_filepath = tmp_path / "test_model.pkl"
        true_model.result = [1, 2, 3, 4]

        write_salt_model(true_model, model_filepath)

        assert model_filepath.exists()

    def test__models_save_and_load(self, true_model, tmp_path):

        true_model.result = [1, 2, 3, 4]

        model_filepath = tmp_path / "test_model.pkl"
        write_salt_model(true_model, model_filepath)

        recov_model = read_salt_model(model_filepath)
        assert isinstance(recov_model, sncosmo.Model)

    def test__model_save_no_result(self, true_model, tmp_path):
        model_filepath = tmp_path / "test_model.pkl"
        assert not hasattr(true_model, "result")

        write_salt_model(true_model, model_filepath)
        assert model_filepath.exists()

    def test__model_load_no_result(self, true_model, tmp_path):
        model_filepath = tmp_path / "test_model.pkl"
        assert not hasattr(true_model, "result")

        write_salt_model(true_model, model_filepath)

        recov_model = read_salt_model(model_filepath)

        assert isinstance(recov_model, sncosmo.Model)
