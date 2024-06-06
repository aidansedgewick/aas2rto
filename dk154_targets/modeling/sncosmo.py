import traceback
import warnings
from logging import getLogger
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.table import Table

import sncosmo

try:
    import iminuit
except ModuleNotFoundError as e:
    msg = (
        "no module iminuit, which `sncosmo` needs"
        "\n    try \033[33;1mpython3 -m pip install iminuit\033[0m"
    )
    raise ModuleNotFoundError(msg)

from dustmaps import sfd

from dk154_targets.target import Target
from dk154_targets.utils import check_unexpected_config_keys


from dk154_targets import paths

logger = getLogger("sncosmo_model")


try:
    sfdq = sfd.SFDQuery()
    sfdq_traceback = None
except FileNotFoundError as e:
    sfdq = None
    sfdq_traceback = traceback.format_exc()

    init_filepath = paths.base_path / "scripts/init_sfd_maps.py"
    try:
        relpath = init_filepath.relative_to(Path.cwd())
    except Exception as e:
        relpath = "scripts/init_sfd_maps.py"

    err_msg = sfdq_traceback + f"\n    try: \033[33;1mpython3 {relpath}\033[0m"
    raise FileNotFoundError(err_msg)


def build_astropy_lightcurve(detections: pd.DataFrame) -> Table:
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


def get_detections(
    lc: pd.DataFrame, use_badqual=True, valid_tag="valid", badqual_tag="badqual"
):

    if "tag" in lc.columns:
        if use_badqual:
            data = lc[np.isin(lc["tag"], [valid_tag, badqual_tag])]
        else:
            data = lc[lc["tag"] == "valid"]
        return data
    else:
        return lc


def initialise_model() -> sncosmo.Model:
    dust = sncosmo.F99Dust()
    model = sncosmo.Model(
        source="salt2", effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )
    return model


def write_emcee_salt_model():
    pass


def read_emcee_salt_model():
    pass


class SncosmoSaltModeler:

    default_nsamples = 1500
    default_nwalkers = 12

    def __init__(
        self,
        faint_limit=99.0,
        use_badqual=True,
        min_detections=3,
        initializer=None,
        existing_models_path=None,
        show_traceback=True,
        use_emcee=True,
        nsamples=None,
        nwalkers=None,
        **kwargs,
    ):
        self.__name__ = "sncosmo_salt"
        self.faint_limit = faint_limit
        self.use_badqual = use_badqual
        self.min_detections = min_detections

        if existing_models_path is not None:
            existing_models_path = Path(existing_models_path)
            existing_models_path.mkdir(exist_ok=True, parent=True)
        self.existing_models_path = existing_models_path

        self.show_traceback = show_traceback
        self.use_emcee = use_emcee
        self.initializer = initializer
        self.nsamples = nsamples or self.default_nsamples
        self.nwalkers = nwalkers or self.default_nwalkers

        logger.info
        logger.info(f"set use_emcee: {self.use_emcee}")
        if self.use_emcee:
            logger.info(f"nsamples={self.nsamples}, nwalkers={self.nwalkers}")
        logger.info(f"set faint_limit={self.faint_limit} (no models for fainter)")
        logger.info(f"set use_badqual: {self.use_badqual}")

        for k, v in kwargs.items():
            logger.warning(f"\033[33;1munknown kwarg\033[0m {k}={v}")

    def __call__(self, target: Target):
        if sncosmo is None:
            msg = (
                "`sncosmo` not imported properly. try:\n    "
                "\033[33;1mpython3 -m pip install sncosmo\033[0m"
            )
            raise ModuleNotFoundError(msg)

        detections = get_detections(
            target.compiled_lightcurve, use_badqual=self.use_badqual
        )
        lightcurve = build_astropy_lightcurve(detections)

        brightest_detection = detections["mag"].min()
        if brightest_detection > self.faint_limit:
            msg = f"brightest {brightest_detection} > {self.faint_limit}: too faint!"
            logger.info(msg)
            return None

        N_detections = {}
        for fid, fid_history in detections.groupby("band"):
            N_detections[fid] = len(fid_history)

        # N_ztfg = N_detections.get("ztfg", 0)
        # N_ztfr = N_detections.get("ztfr", 0)
        # enough_detections = (N_ztfg > 1 and N_ztfr > 1) or (N_ztfg + N_ztfr > 2)
        # if not enough_detections:
        #     logger.debug(f"{target.objectId} too few detections:\n    {N_detections}")
        #     return
        if len(detections) < self.min_detections:
            return

        if sfdq is None:
            print(sfdq_traceback)
            msg = (
                "sfd.SDFQuery() not initialised properly. try:\n     "
                "python3 scripts/init_sfd_maps.py"
            )
            logger.warning(msg)
            return

        if self.initializer is None:
            model = initialise_model()
            mwebv = sfdq(target.coord)
            model.set(mwebv=mwebv)

            fitting_params = model.param_names
            fitting_params.remove("mwebv")
        else:
            model = self.initializer()
            fitting_params = model.param_names

        # known_redshift = target.tns_data.parameters.get("Redshift", None)
        # if known_redshift is not None and np.isfinite(known_redshift):

        fitting_params = model.param_names

        known_redshift = None
        tns_data = target.target_data.get("tns", None)
        if tns_data is not None:
            if tns_data.parameters is not None:
                known_redshift = tns_data.parameters.get("Redshift", None)

        if known_redshift is not None:
            logger.debug(f"{target.objectId} use known TNS z={known_redshift:.3f}")
            model.set(z=known_redshift)
            fitting_params.remove("z")
            bounds = {}
        else:
            bounds = dict(z=(0.001, 0.2))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lsq_result, lsq_fitted_model = sncosmo.fit_lc(
                    lightcurve, model, fitting_params, bounds=bounds
                )
                if self.use_emcee:
                    try:
                        result, fitted_model = sncosmo.mcmc_lc(
                            lightcurve,
                            lsq_fitted_model,
                            fitting_params,
                            nsamples=self.nsamples,
                            nwalkers=self.nwalkers,
                            bounds=bounds,
                        )
                    except Exception as e:
                        errname = type(e).__name__
                        msg = f"{target.objectId} mcmc: {errname}\n    {e}"
                        logger.warning(msg)
                        if self.show_traceback:
                            tr = traceback.format_exc()
                            print(tr)
                        fitted_model = lsq_fitted_model
                        result = lsq_result
                else:
                    fitted_model = lsq_fitted_model
                    result = lsq_result
            fitted_model.result = result
            logger.debug(f"{target.objectId} fitted model!")

        except Exception as e:
            errname = type(e).__name__
            msg = f"{target.objectId} lsq: {errname}\n    {e}"
            logger.warning(msg)
            if self.show_traceback:
                tr = traceback.format_exc()
                print(tr)
            fitted_model = None
        return fitted_model
