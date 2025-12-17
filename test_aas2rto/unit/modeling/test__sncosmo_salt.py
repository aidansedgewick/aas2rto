import pickle
import pytest
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.table import Table
from astropy.time import Time

import sncosmo

from dustmaps.sfd import SFDQuery

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import Table

from aas2rto.modeling.sncosmo_salt import (
    build_input_lightcurve,
    get_detections,
    PickleableF99Dust,
    initialize_model,
    initialize_pickleable_model,
    write_salt_model,
    read_salt_model,
    SncosmoSaltModeler,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def input_detections(lc_pandas: pd.DataFrame):
    return get_detections(lc_pandas)


@pytest.fixture
def t_model():
    return Time(60020.0, format="mjd")


@pytest.fixture(scope="module")
def lockman_coord():
    # The Lockman Hole, minimum MW dust column density
    return SkyCoord(ra="10:34:00", dec="+57:40:00", unit=(u.hourangle, u.deg))


@pytest.fixture(scope="module")
def lockman_ebv(lockman_coord: SkyCoord):
    sfdq = SFDQuery()
    ebv = sfdq.query(lockman_coord)
    return ebv


@pytest.fixture
def salt_model(lockman_ebv: float):
    model = initialize_model()
    model.set(mwebv=lockman_ebv)
    z = 0.02
    x1 = 0.1
    c = -0.05
    MB = -19.3
    mB = Planck18.distmod(z).value + MB - 0.148 * x1 + 3.1 * c
    x0 = 10 ** (-0.4 * (mB - 10.5))
    model.set(z=z, t0=60010.0, x0=x0, x1=x1, c=c)
    return model


@pytest.fixture
def result_with_samples(salt_model: sncosmo.Model):
    mu_vals = [salt_model[p] for p in "z x0 x1 c t0".split()]
    sig_vals = [0.005, 1e-4, 0.05, 0.01, 0.3]
    samples = np.column_stack(
        [np.random.normal(mu, sig, 500) for mu, sig in zip(mu_vals, sig_vals)]
    )
    return {"ndof": 5, "chisq": 5.0, "samples": samples}


@pytest.fixture
def fittable_lc(salt_model: sncosmo.Model):
    print(salt_model.parameters)
    ztfg_t_grid = np.arange(60000.0, 60020.0, 2.0)
    N_ztfg = len(ztfg_t_grid)
    ztfg_m_grid = salt_model.bandmag("ztfg", "ab", ztfg_t_grid)
    ztfr_t_grid = np.arange(60001.0, 60021.0, 2.0)
    N_ztfr = len(ztfr_t_grid)
    ztfr_m_grid = salt_model.bandmag("ztfr", "ab", ztfr_t_grid)

    concat_data = dict(
        mjd=np.concat([ztfg_t_grid, ztfr_t_grid]),
        mag=np.concat([ztfg_m_grid, ztfr_m_grid]),
        band=np.concat([["ztfg"] * N_ztfg, ["ztfr"] * N_ztfr]),
    )

    lc = pd.DataFrame(concat_data)
    lc.sort_values("mjd", inplace=True, ignore_index=True)

    lc.loc[:, "magerr"] = 0.1
    lc.loc[:, "valid"] = "tag"
    return lc


@pytest.fixture
def target_to_model(
    lockman_coord: SkyCoord,
    fittable_lc: pd.DataFrame,
):
    target = Target("T00m", lockman_coord)
    target.compiled_lightcurve = fittable_lc.copy()
    return target


@pytest.fixture
def salt_modeler():
    return SncosmoSaltModeler(use_emcee=False)


class Test__GetDetectionsHelper:
    def test__get_detections(self, lc_pandas: pd.DataFrame):
        # Act
        result = get_detections(lc_pandas)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test__get_det_no_bad_qual(self, lc_pandas: pd.DataFrame):
        # Act
        result = get_detections(lc_pandas, use_badqual=False)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6


class Test__InputLCHelper:
    def test__build_input_lc(self, input_detections: pd.DataFrame):
        # Act
        input_lc = build_input_lightcurve(input_detections)

        # Assert
        exp_cols = "time band mag magerr flux fluxerr snr zp zpsys".split()
        assert set(input_lc.columns) == set(exp_cols)


class Test__PickleableF99Dust:
    def test__pickle_f99(self, tmp_path: Path):
        # Arrange
        output_path = tmp_path / "pickleable_f99.pkl"
        f99 = PickleableF99Dust()

        # Act
        with open(output_path, "wb+") as f:
            pickle.dump(f99, f)

        # Assert
        with open(output_path, "rb") as f:
            f99_rec = pickle.load(f)
        assert isinstance(f99_rec, PickleableF99Dust)

    def test__propagate(self):
        # Arrange
        f99_pkl = PickleableF99Dust()
        f99_snc = sncosmo.F99Dust()
        wv_grid = np.arange(1000.0, 10000.0, 10.0)
        f_grid = np.ones(len(wv_grid))

        # Act
        ext_grid_pkl = f99_pkl.propagate(wv_grid, f_grid)
        ext_grid_snc = f99_snc.propagate(wv_grid, f_grid)

        # Assert
        assert np.allclose(ext_grid_pkl, ext_grid_snc)


class Test__ModelInitHelpers:
    def test__model_init(self):
        # Act
        model = initialize_model()

        # Assert
        assert isinstance(model, sncosmo.Model)
        assert isinstance(model.effects[0], sncosmo.F99Dust)

    def test__model_no_pkl(self, tmp_path: Path):
        # Arrange
        model = initialize_model()
        model_path = tmp_path / "model.pkl"

        # Act
        with pytest.raises(Exception):
            with open(model_path, "wb+") as f:
                pickle.dump(model, f)

        # Assert
        with pytest.raises(EOFError):
            with open(model_path, "rb") as f:
                recover = pickle.load(f)

    def test__pickleable_model_init(self):
        # Act
        model = initialize_pickleable_model()

        # Assert
        assert isinstance(model, sncosmo.Model)
        assert isinstance(model.effects[0], PickleableF99Dust)

    def test__pickleable_model_pkl(self, tmp_path: Path):
        # Arrange
        model = initialize_pickleable_model()
        model_path = tmp_path / "model.pkl"

        # Act
        with open(model_path, "wb+") as f:
            pickle.dump(model, f)

        # Assert
        assert model_path.exists()
        with open(model_path, "rb") as f:
            rec_model = pickle.load(f)
        assert isinstance(rec_model, sncosmo.Model)
        assert isinstance(model.effects[0], PickleableF99Dust)


class Test__WriteSaltModel:
    def test__write_salt(self, salt_model: sncosmo.Model, tmp_path: Path):
        # Arrange
        salt_model.result = {"ndof": 5, "chisq": 5.0}  # what a nice fit!
        model_path = tmp_path / "T00m_sncosmo_salt.pkl"

        # Act
        write_salt_model(salt_model, model_path)

        # Assert
        assert model_path.exists()
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, dict)
        assert set(data.keys()) == set(["parameters", "result"])

        exp_params = "z x0 x1 c t0 mwebv".split()
        assert isinstance(data["parameters"], dict)
        assert set(data["parameters"].keys()) == set(exp_params)


class Test__ReadSaltModel:
    def test__read_salt(self, salt_model: sncosmo.Model, tmp_path: Path):
        # Arrange
        salt_model.result = {"ndof": 5, "chisq": 5.0}  # what a nice fit!
        model_path = tmp_path / "T00m_sncosmo_salt.pkl"
        write_salt_model(salt_model, model_path)

        # Act
        recovered = read_salt_model(model_path, initialize_model)

        # Assert
        assert isinstance(recovered, sncosmo.Model)
        assert np.isclose(recovered["z"], 0.02)
        assert np.isclose(recovered["x1"], 0.1)
        assert np.isclose(recovered["c"], -0.05)
        assert np.isclose(recovered["t0"] - 60000.0, 10.0)

    def test__read_salt_with_samples(
        self, salt_model: sncosmo.Model, result_with_samples: dict, tmp_path: Path
    ):
        # Arrange
        salt_model.result = result_with_samples
        model_path = tmp_path / "T00m_sncosmo_salt.pkl"
        write_salt_model(salt_model, model_path)

        # Act
        recovered = read_salt_model(model_path, initialize_model)

        # Assert
        assert len(recovered.result["samples"]) == 500


class Test__FittableLC:
    def test__lc(self, fittable_lc: pd.DataFrame):
        assert len(fittable_lc) == 20
        assert fittable_lc["band"].iloc[0] == "ztfg"
        assert fittable_lc["band"].iloc[1] == "ztfr"
        assert all(fittable_lc["mag"] < 23.0)
        assert all(fittable_lc["mag"] > 10.0)

    def test__input_from_lc(self, fittable_lc: pd.DataFrame):
        # Act
        # Check the modeling won't fail simply because of a bad lighcure
        input_lc = build_input_lightcurve(fittable_lc)

        # Assert
        assert isinstance(input_lc, Table)


class Test__ModelerInit:
    def test__salt_modeler_init(self):
        # Act
        modeler = SncosmoSaltModeler()

        # Assert
        assert isinstance(modeler.faint_limit, float)
        assert isinstance(modeler.use_badqual, bool)
        assert isinstance(modeler.min_detections, int)

        assert modeler.existing_models_path is None

        assert isinstance(modeler.model_key, str)
        assert callable(modeler.initializer)
        assert isinstance(modeler.use_emcee, bool)
        assert isinstance(modeler.nsamples, int)
        assert isinstance(modeler.nwalkers, int)
        assert isinstance(modeler.z_max, float)

        assert isinstance(modeler.show_traceback, bool)

    def test__init_with_path(self, tmp_path: Path):
        # Arrange
        models_path = tmp_path / "models"
        assert not models_path.exists()

        # Act
        modeler = SncosmoSaltModeler(existing_models_path=models_path)

        # Assert
        assert isinstance(modeler.existing_models_path, Path)
        assert models_path.exists()
        assert models_path.is_dir()

    def test__bad_sfdq_raises(self, monkeypatch):
        # Act
        with monkeypatch.context() as mp:
            mp.setattr("aas2rto.modeling.sncosmo_salt.sfdq", None)
            with pytest.raises(FileNotFoundError):
                modeler = SncosmoSaltModeler()


class Test__ModelTarget:
    def test__model_target(
        self, salt_modeler: SncosmoSaltModeler, target_to_model: Target, t_model: Time
    ):
        # Arrange
        assert not salt_modeler.use_emcee

        # Act
        fitted_model = salt_modeler(target_to_model, t_ref=t_model)

        # Assert
        assert isinstance(fitted_model, sncosmo.Model)
        assert np.isclose(fitted_model["z"], 0.02, rtol=0.1, atol=0.01)
        # How to check mB/x0...?
        assert np.isclose(fitted_model["x1"], 0.1, rtol=0.1, atol=0.05)
        assert np.isclose(fitted_model["c"], -0.05, rtol=0.1, atol=0.01)

        assert isinstance(fitted_model.result, dict)

    def test__no_lc_no_fail(
        self, salt_modeler: SncosmoSaltModeler, target_to_model: Target, t_model: Time
    ):
        # Arrange
        target_to_model.compiled_lightcurve = None

        # Act
        fitted_model = salt_modeler(target_to_model, t_ref=t_model)

        # Assert
        assert fitted_model is None

    def test__with_fixed_redshift(
        self,
        salt_modeler: SncosmoSaltModeler,
        target_to_model: Target,
        tns_td: TargetData,
        t_model: Time,
    ):
        # Arrange
        target_to_model.target_data["tns"] = tns_td

        # Act
        fitted_model = salt_modeler(target_to_model, t_ref=t_model)

        # Assert
        assert np.isclose(fitted_model["z"], 0.02)

    def test__no_lc_no_model(
        self, salt_modeler: SncosmoSaltModeler, target_to_model: Target, t_model: Time
    ):
        # Arrange
        target_to_model.compiled_lightcurve = None

        # Act
        fitted_model = salt_modeler(target_to_model, t_ref=t_model)

        # Assert
        assert fitted_model is None

    def test__too_faint_no_model(self, target_to_model: Target, t_model: Time):
        # Arrange
        modeler = SncosmoSaltModeler(faint_limit=12.0)
        assert all(target_to_model.compiled_lightcurve["mag"] > 12.0)

        # Act
        result = modeler(target_to_model, t_ref=t_model)

        # Assert
        assert result is None

    def test__too_few_dets_no_model(self, target_to_model: Target, t_model: Time):
        # Arrange
        modeler = SncosmoSaltModeler(min_detections=25)
        assert len(target_to_model.compiled_lightcurve) < 25

        # Act
        fitted_model = modeler(target_to_model, t_ref=t_model)

        # Assert
        assert fitted_model is None

    def test__other_initializer(self, target_to_model: Target, t_model: Time):
        # Arrange
        modeler = SncosmoSaltModeler(initializer=initialize_pickleable_model)

        # Act
        fitted_model = modeler(target_to_model, t_ref=t_model)

        # Assert
        assert isinstance(fitted_model, sncosmo.Model)

    def test__bad_dataqual_no_fail(
        self, salt_modeler: SncosmoSaltModeler, target_to_model: Target, t_model: Time
    ):
        # Arrange
        target_to_model.compiled_lightcurve["magerr"] = 1.0  # everything S/N=1
        input_lc = build_input_lightcurve(target_to_model.compiled_lightcurve)
        model = initialize_model()

        # Act
        with pytest.raises(sncosmo.fitting.DataQualityError):
            _, _ = sncosmo.fit_lc(
                input_lc, model, ["z", "x0", "x1", "c", "t0"], bounds={"z": (0.0, 0.1)}
            )
        fitted_model = salt_modeler(target_to_model, t_ref=t_model)

        # Assert
        assert fitted_model is None

    def test__write_model(self, target_to_model: Target, t_model: Time, tmp_path: Path):
        # Arrange
        existing_models_path = tmp_path / "models"
        modeler = SncosmoSaltModeler(existing_models_path=existing_models_path)

        # Act
        model = modeler(target_to_model, t_ref=t_model)

        # Assert
        exp_model_path = tmp_path / "models/T00m_sncosmo_salt.pkl"
        assert exp_model_path.exists()

        with open(exp_model_path, "rb") as f:
            data = pickle.load(f)

        assert isinstance(data, dict)
        assert set(data.keys()) == set(["parameters", "result"])

    def test__read_if_missing_model(
        self,
        salt_model: sncosmo.Model,
        target_to_model: Target,
        t_model: Time,
        tmp_path: Path,
    ):
        # Arrange
        salt_model.result = {"weird_key": "this key would never be in a model!"}
        existing_models_path = tmp_path / "models"
        existing_models_path.mkdir(exist_ok=True, parents=True)
        T00m_model_path = existing_models_path / "T00m_sncosmo_salt.pkl"
        write_salt_model(salt_model, T00m_model_path)
        modeler = SncosmoSaltModeler(existing_models_path=existing_models_path)

        # Act
        fitted_model = modeler(target_to_model, t_ref=t_model)

        # Assert
        assert "weird_key" in fitted_model.result.keys()

    def test__no_read_if_existing_models(
        self,
        salt_model: sncosmo.Model,
        target_to_model: Target,
        t_model: Time,
        tmp_path: Path,
    ):
        # Arrange
        salt_model.result = {"weird_key": "this key would never be in a model!"}
        existing_models_path = tmp_path / "models"
        existing_models_path.mkdir(exist_ok=True, parents=True)
        T00m_model_path = existing_models_path / "T00m_sncosmo_salt.pkl"
        write_salt_model(salt_model, T00m_model_path)
        # Imagine we recovered it last time
        target_to_model.models["sncosmo_salt"] = salt_model

        modeler = SncosmoSaltModeler(existing_models_path=existing_models_path)

        # Act
        fitted_model = modeler(target_to_model, t_ref=t_model)

        # Assert
        assert "weird_key" not in fitted_model.result.keys()
        assert "ndof" in fitted_model.result.keys()  # eg. the normal stuff.


class Test__EmceeModeler:
    def test__emcee_model(self, target_to_model: Target, t_model: Time):
        # Arrange
        modeler = SncosmoSaltModeler(use_emcee=True, nsamples=250, nwalkers=12)

        # Act
        fitted_model = modeler(target_to_model)

        # Assert
        assert isinstance(fitted_model, sncosmo.Model)
        assert "samples" in fitted_model.result.keys()

    def test__bad_emcee_params_no_fail(self, target_to_model: Target, t_model: Time):
        # Arrange
        input_lc = build_input_lightcurve(target_to_model.compiled_lightcurve)
        model = initialize_model()
        fitting_params = ["z", "x0", "x1", "c", "t0"]
        bounds = {"z": (0.0, 0.1)}
        lsq_res, lsq_model = sncosmo.fit_lc(
            input_lc, model, fitting_params, bounds=bounds
        )
        modeler = SncosmoSaltModeler(
            use_emcee=True, show_traceback=True, nsamples="what", nwalkers="blah"
        )

        # Act
        with pytest.raises(Exception):
            _, _ = sncosmo.mcmc_lc(
                input_lc,
                lsq_model,
                fitting_params,
                nsamples="what",  # These parameters will cause a crash!
                nwalkers="blah",  # ""    ""
                bounds=bounds,
            )
        fitted_model = modeler(target_to_model, t_ref=t_model)
