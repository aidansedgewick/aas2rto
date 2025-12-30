import copy
import pickle
import traceback
import warnings
from logging import getLogger
from pathlib import Path
from typing import Callable

import numpy as np

import pandas as pd

import astropy.units as u
from astropy.table import Table
from astropy.time import Time

try:
    import sncosmo
except ModuleNotFoundError as e:
    msg = (
        "`sncosmo` not imported properly. try:\n    "
        "\033[33;1mpython3 -m pip install sncosmo\033[0m"
    )
    raise ModuleNotFoundError(msg)

try:
    import iminuit
except ModuleNotFoundError as e:
    msg = (
        "no module iminuit, which `sncosmo` needs"
        "\n    try \033[33;1mpython3 -m pip install iminuit\033[0m"
    )
    raise ModuleNotFoundError(msg)

from dustmaps import sfd

from dust_extinction.parameter_averages import F99

from aas2rto import paths
from aas2rto.target import Target


logger = getLogger(__name__.split(".")[-1])

try:
    sfdq = sfd.SFDQuery()
    sfdq_traceback = None
except FileNotFoundError as e:
    sfdq = None
    sfdq_traceback = traceback.format_exc()


def get_detections(
    lc: pd.DataFrame, use_badqual=True, valid_tag="valid", badqual_tag="badqual"
):

    if "tag" in lc.columns:
        if use_badqual:
            data = lc[np.isin(lc["tag"], [valid_tag, badqual_tag])]
        else:
            data = lc[lc["tag"] == valid_tag]
        return data
    else:
        return lc


def build_input_lightcurve(detections: pd.DataFrame) -> Table:
    data = dict(
        time=detections["mjd"].values,  # .values is an np array...
        # band=detections["band"].map(ztf_band_lookup).values,
        band=detections["band"].values,
        mag=detections["mag"].values,
        magerr=detections["magerr"].values,
    )
    lc = Table(data)
    lc["flux"] = 10 ** (0.4 * (8.9 - lc["mag"]))
    lc["fluxerr"] = lc["flux"] * lc["magerr"] * np.log(10.0) / 2.5
    lc["snr"] = (np.log(10.0) / 2.5) / lc["magerr"]
    lc["zp"] = np.full(len(lc), 8.9)
    lc["zpsys"] = np.full(len(lc), "ab")
    return lc


class PickleableF99Dust(sncosmo.PropagationEffect):

    _minwave = 909.09
    _maxwave = 60000.0

    def __init__(self, r_v=3.1):
        self._param_names = ["ebv"]
        self.param_names_latex = ["E(B-V)"]
        self._parameters = np.array([0.0])
        self._r_v = r_v
        self._f = None

    def propagate(self, wave, flux, phase=None):
        ebv = self._parameters[0]

        inv_wave = 1e4 / wave  # per micron
        axav = F99.evaluate(inv_wave, Rv=self._r_v)

        av = self._r_v * ebv
        ext = np.power(10.0, -0.4 * axav * av)
        return ext * flux


def initialize_model() -> sncosmo.Model:
    dust = sncosmo.F99Dust()  #
    model = sncosmo.Model(
        source="salt3", effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )
    return model


def initialize_pickleable_model() -> sncosmo.Model:
    dust = PickleableF99Dust()
    model = sncosmo.Model(
        source="salt3", effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )
    return model


def write_salt_model(model: sncosmo.Model, filepath: Path):
    parameters = {k: v for k, v in zip(model.param_names, model.parameters)}
    model_result = getattr(model, "result", None)

    model_data = {"parameters": parameters, "result": model_result}

    # CANNOT just pickle whole model: sncosmo.F99Dust() is not serializable.
    with open(filepath, "wb+") as f:
        pickle.dump(model_data, f)
    return


def read_salt_model(filepath: Path, initializer: Callable):
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)

    model = initializer()
    model.set(**model_data["parameters"])
    model.result = model_data.get("result", None)
    return model


class SncosmoSaltModeler:

    def __init__(
        self,
        model_key: str = None,
        faint_limit: float = 99.0,
        use_badqual: bool = True,
        min_detections: int = 3,
        set_mw_ebv: bool = True,
        existing_models_path: Path = None,
        initializer: Callable = None,
        use_emcee: bool = False,
        nsamples: int = 2500,
        nwalkers: int = 12,
        z_max: float = 0.2,
        show_traceback: bool = True,
    ):

        self.__name__ = "sncosmo_salt"
        self.model_key = model_key or self.__name__

        self.faint_limit = faint_limit
        self.use_badqual = use_badqual
        self.min_detections = min_detections

        self.set_mw_ebv = set_mw_ebv
        if self.set_mw_ebv:
            if sfdq is None:
                logger.error(sfdq_traceback)
                msg = (
                    "sfd.SDFQuery() not initialised properly. try:\n     "
                    "python3 scripts/init_sfd_maps.py"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

        self.initializer = initializer or initialize_model
        self.use_emcee = use_emcee
        self.nsamples = nsamples
        self.nwalkers = nwalkers
        self.z_max = z_max

        if existing_models_path is not None:
            existing_models_path = Path(existing_models_path)
            existing_models_path.mkdir(exist_ok=True, parents=True)
        self.existing_models_path = existing_models_path

        self.show_traceback = show_traceback

        logger.info(f"set use_emcee: {self.use_emcee}")
        if self.use_emcee:
            logger.info(f"nsamples={self.nsamples}, nwalkers={self.nwalkers}")
        logger.info(f"set faint_limit={self.faint_limit} (no models for fainter)")
        logger.info(f"set use_badqual: {self.use_badqual}")

    def __call__(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        target_id = target.target_id

        existing_model = target.models.get(self.model_key, None)

        ## ===== Can we just load an existing model from file?? ===== ##
        if self.existing_models_path is not None:
            model_stem = f"{target_id}_{self.model_key}"
            model_filepath = self.existing_models_path / f"{model_stem}.pkl"
            if existing_model is None and model_filepath.exists():
                try:
                    model = read_salt_model(model_filepath, self.initializer)
                except Exception as e:
                    logger.error(f"{target_id}: error loading {model_filepath.stem}")
                    if self.show_traceback:
                        tr = traceback.format_exc()
                        logger.error(f"TRACEBACK:\n{tr}")
                    model = None
                if model is not None:
                    return model
        else:
            model_filepath = None

        ##===== Is the data sufficient to build a model?? =====##
        if target.compiled_lightcurve is None:
            return None

        detections = get_detections(
            target.compiled_lightcurve, use_badqual=self.use_badqual
        )
        input_lightcurve = build_input_lightcurve(detections)

        brightest_detection = detections["mag"].min()
        if brightest_detection > self.faint_limit:
            msg = (
                f"no model for {target_id}:\n    "
                f"brightest {brightest_detection} > {self.faint_limit}: too faint!"
            )
            logger.debug(msg)
            return None

        N_detections = {}
        for fid, fid_history in detections.groupby("band"):
            N_detections[fid] = len(fid_history)

        if len(detections) < self.min_detections:
            logger.debug(
                f"{target_id}: too few detections "
                f"{len(detections)} < {self.min_detections} (min)"
            )
            return None

        ##===== Now actually start to build the model =====##
        model: sncosmo.Model = self.initializer()
        fitting_params = model.param_names

        if self.set_mw_ebv:
            mwebv = sfdq(target.coord)
            model.set(mwebv=mwebv)

            fitting_params = model.param_names
            fitting_params.remove("mwebv")

        fitting_params = model.param_names

        ##===== Do we already know the redshift?? =====##
        bounds = {"z": (0.001, self.z_max)}
        tns_data = target.target_data.get("tns", None)
        if tns_data is not None:
            known_redshift = float(tns_data.parameters.get("redshift", "nan"))
            if np.isfinite(known_redshift):
                logger.debug(f"{target.target_id} use known TNS z={known_redshift:.3f}")
                model.set(z=known_redshift)
                fitting_params.remove("z")
                bounds.pop("z")

        ##===== Actually fit the model now!! =====##
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lsq_result, lsq_fitted_model = sncosmo.fit_lc(
                    input_lightcurve, model, fitting_params, bounds=bounds
                )

                ##===== Do we want to do MCMC?? =====##
                if self.use_emcee:
                    try:
                        result, fitted_model = sncosmo.mcmc_lc(
                            input_lightcurve,
                            lsq_fitted_model,
                            fitting_params,
                            nsamples=self.nsamples,
                            nwalkers=self.nwalkers,
                            bounds=bounds,
                        )
                    except Exception as e:
                        errname = type(e).__name__
                        msg = f"{target.target_id} mcmc error: {errname}\n    {e}"
                        logger.warning(msg)
                        if self.show_traceback:
                            tr = traceback.format_exc()
                            logger.error(f"TRACEBACK:\n{tr}")
                        fitted_model = lsq_fitted_model
                        result = lsq_result
                else:
                    fitted_model = lsq_fitted_model
                    result = lsq_result
            fitted_model.result = result
            logger.debug(f"{target.target_id} fitted model!")

        except Exception as e:
            errname = type(e).__name__
            msg = f"{target.target_id} lsq error: {errname}\n    {e}"
            logger.warning(msg)
            if self.show_traceback and type(e).__name__ not in [
                "DataQualityError",
                "LinAlgError",
            ]:
                tr = traceback.format_exc()
                logger.error(f"TRACEBACK:\n{tr}")
            fitted_model = None

        ##===== Should we write this model back out again?? =====##
        if model_filepath is not None:
            try:
                write_salt_model(model, model_filepath)
            except Exception as e:
                msg = f"could not pickle data from {type(model)} into {model_filepath}"
                logger.error(msg)
                if self.show_traceback:
                    tr = traceback.format_exc()
                    logger.error(f"TRACEBACK:\n{tr}")

        return fitted_model
