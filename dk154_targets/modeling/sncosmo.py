import traceback
import warnings
from logging import getLogger
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.table import Table

try:
    import sncosmo
except ModuleNotFoundError as e:
    sncosmo = None

try:
    import iminuit
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "\n    try \033[33;1mpython3 -m pip install iminuit\033[0m"
    )

from dustmaps import sfd

from dk154_targets import Target

from dk154_targets import paths

logger = getLogger("sncosmo_model")


try:
    sfdq = sfd.SFDQuery()
    sfdq_traceback = None
except FileNotFoundError as e:
    sfdq = None
    sfdq_traceback = traceback.format_exc()

    init_path = paths.base_path / "scripts/init_sfd_maps.py"
    try:
        relpath = init_path.relative_to(Path.cwd())
    except Exception as e:
        relpath = "scripts/init_sfd_maps.py"

    err_msg = sfdq_traceback + f"\n    try: \033[33;1mpython3 {relpath}\033[0m"
    raise FileNotFoundError(err_msg)


def get_detections(target: Target, use_badquality=True):
    if "tag" in target.compiled_lightcurve.columns:
        if use_badquality:
            tag_query = "(tag=='valid') or (tag=='badquality')"
        else:
            tag_query = "tag=='valid'"

        return target.compiled_lightcurve.query(tag_query)
    else:
        return target.compiled_lightcurve


def build_astropy_lightcurve(detections: pd.DataFrame) -> Table:
    data = dict(
        time=detections["jd"].values,  # .values is an np array...
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


def initialise_model() -> sncosmo.Model:
    dust = sncosmo.F99Dust()
    model = sncosmo.Model(
        source="salt2", effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )
    return model


class sncosmo_salt:
    def __init__(self, use_emcee=True, faint_limit=99.0, use_badquality=True):
        self.__name__ = self.__class__.__name__
        self.use_emcee = use_emcee
        self.faint_limit = faint_limit
        self.use_badquality = use_badquality

        logger.info(f"set use_emcee: {self.use_emcee}")
        logger.info(f"set faint_limit={self.faint_limit} (no models for fainter)")
        logger.info(f"set use_badquality: {self.use_badquality}")

    def __call__(self, target: Target):
        if sncosmo is None:
            msg = (
                "`sncosmo` not imported properly. try:\n    "
                "\033[33;1mpython3 -m pip install sncosmo\033[0m"
            )
            raise ModuleNotFoundError(msg)

        detections = get_detections(target, use_badquality=self.use_badquality)
        lightcurve = build_astropy_lightcurve(detections)

        if detections["mag"].min() > self.faint_limit:
            return None

        N_detections = {}
        for fid, fid_history in target.fink_data.detections.groupby("fid"):
            N_detections[fid] = len(fid_history)

        enough_detections = N_detections.get(1, 0) > 1 and N_detections.get(2, 0) > 1
        if not enough_detections:
            return

        model = initialise_model()

        if sfdq is None:
            print(sfdq_traceback)
            msg = "sfd.SDFQuery() not initialised properly. try:\n     scripts/init_sfd_maps.py"

        mwebv = sfdq(target.coord)
        model.set(mwebv=mwebv)

        fitting_params = model.param_names
        fitting_params.remove("mwebv")

        known_redshift = target.tns_data.parameters.get("Redshift", None)
        if known_redshift is not None and np.isfinite(known_redshift):
            logger.debug(f"{target.objectId} use known TNS z={known_redshift:.3f}")
            model.set(z=known_redshift)
            fitting_params.remove("z")
            bounds = {}
        else:
            bounds = dict(z=(0.001, 0.2))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lsq_result, lsq_fitted_model = sncosmo.fit_lc(
                    lightcurve, model, fitting_params, bounds=bounds
                )
                existing_models = target.models
                if self.use_emcee:
                    try:
                        result, fitted_model = sncosmo.mcmc_lc(
                            lightcurve,
                            lsq_fitted_model,
                            fitting_params,
                            nsamples=1500,
                            nwalkers=12,
                            bounds=bounds,
                        )
                    except Exception as e:
                        tr = traceback.format_exc()
                        logger.warning(f"{target.objectId} during emcee fitting")
                        print(tr)
                        fitted_model = lsq_fitted_model
                        result = lsq_result
                    # else:
                    #    logger.debug"{target.objectId} first attempt: lsq is good enough!")
                    #    result, fitted_model = lsq_result, lsq_fitted_model
                else:
                    fitted_model = lsq_fitted_model
                    result = lsq_result
            fitted_model.result = result

        except Exception as e:
            logger.warning(f"{target.objectId} sncosmo fitting failed")
            tr = traceback.format_exc()
            print(tr)
            fitted_model = None

        return fitted_model
