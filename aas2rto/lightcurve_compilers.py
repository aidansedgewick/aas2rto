import copy
import time
import warnings
from logging import getLogger

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto import utils
from aas2rto.target import Target
from aas2rto.target_data import TargetData

logger = getLogger(__name__.split(".")[-1])

DEFAULT_VALID_TAG = "valid"
DEFAULT_BADQUAL_TAG = "badqual"
DEFAULT_ULIMIT_TAG = "upperlim"


def _check_is_target_data(
    data, source_name="<unknown data source>", target_id="<unknown target_id>"
):
    if isinstance(data, TargetData):
        return True
    data_type = type(data)
    msg = (
        f"broker '{source_name}' data for {target_id}"
        f"is type {data_type}, not TargetData"
    )
    logger.warning(msg)
    warnings.warn(UserWarning(msg))
    return False


def prepare_ztf_data(
    ztf_data: TargetData,
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):

    data_list = []
    if ztf_data.detections is not None:
        detections = ztf_data.detections.copy()
        detections.loc[:, "tag"] = valid_tag
        # data_list = [detections]
        data_list.append(detections)
    if ztf_data.badqual is not None:
        badqual = ztf_data.badqual.copy()
        badqual.loc[:, "tag"] = badqual_tag
        data_list.append(badqual)
    if ztf_data.non_detections is not None:
        ulimits = ztf_data.non_detections.copy()
        ulimits.loc[:, "tag"] = ulimit_tag
        data_list.append(ulimits)

    # Now combine what's available.
    if len(data_list) > 0:
        ztf_lc = pd.concat(data_list, ignore_index=True)
    else:
        ztf_lc = ztf_data.lightcurve

    ztf_lc["source"] = "ztf"

    # Fix column names here
    ztf_band_lookup = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    ztf_colmap = {"magpsf": "mag", "sigmapsf": "magerr", "candid": "alert_id"}

    # Rename the bands (eg. for sncosmo)
    ztf_lc.loc[:, "band"] = ztf_lc["fid"].map(ztf_band_lookup)
    ztf_lc.rename(ztf_colmap, axis=1, inplace=True)

    ztf_lc.sort_values("mjd", inplace=True)
    use_cols = "mjd mag magerr diffmaglim tag band source alert_id".split()
    return ztf_lc[use_cols]


def prepare_lsst_data(
    lsst_data: TargetData,
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):

    data_list = []
    if lsst_data.detections is not None:
        detections = lsst_data.detections.copy()
        detections.loc[:, "tag"] = valid_tag
        # data_list = [detections]
        data_list.append(detections)
    if lsst_data.badqual is not None:
        badqual = lsst_data.badqual.copy()
        badqual.loc[:, "tag"] = badqual_tag
        data_list.append(badqual)
    if lsst_data.non_detections is not None:
        ulimits = lsst_data.non_detections.copy()
        ulimits.loc[:, "tag"] = ulimit_tag
        data_list.append(ulimits)

    # Now combine what's available
    if len(data_list) > 0:
        lsst_lc = pd.concat(data_list, ignore_index=True)
    else:
        lsst_lc = lsst_data.lightcurve.copy()
        lsst_lc.loc[:, "tag"] = valid_tag

    lsst_lc.loc[:, "source"] = "lsst"

    # Convert fluxes into mags and get the upperlims
    psfFlux_snr = lsst_lc["psfFlux"] / lsst_lc["psfFluxErr"]
    lsst_lc.loc[:, "mag"] = -2.5 * np.log10(lsst_lc["psfFlux"]) + 31.4  # flux in nJy
    lsst_lc.loc[:, "magerr"] = 1.09 / psfFlux_snr  # 2.5 / ln(10) ~ 1.09
    if "sky" in lsst_lc:
        lsst_lc.loc[:, "diffmaglim"] = -2.5 * np.log10(lsst_lc["sky"]) + 31.4
    else:
        lsst_lc.loc[:, "diffmaglim"] = 0.0

    # Fix some column names
    lsst_colmap = {"midpointMjdTai": "mjd", "diaSourceId": "alert_id"}
    lsst_lc.rename(lsst_colmap, axis=1, inplace=True)

    # Rename the bands
    lsst_band_lookup = {b: f"lsst{b}" for b in "u g r i z y".split()}
    lsst_lc.loc[:, "band"] = lsst_lc["band"].map(lsst_band_lookup)

    lsst_lc.sort_values("mjd", inplace=True)
    keep_cols = "mjd mag magerr diffmaglim tag band alert_id source".split()
    return lsst_lc[keep_cols]


def prepare_atlas_data(
    atlas_data: TargetData,
    average_epochs: bool = True,
    rolling_window: float = 0.1,
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):
    atlas_lc = atlas_data.lightcurve.copy()

    # atlas_df["snr"] = atlas_df["uJy"] / atlas_df["duJy"]

    # Convert mags to fluxes for averaging.
    dm_snr = 2.5 / np.log(10.0)
    atlas_lc["snr"] = dm_snr / atlas_lc["dm"]

    flux_vals = 10 ** (-0.4 * (abs(atlas_lc["m"]) - 23.9))
    flux_sign = np.sign(atlas_lc["m"])
    atlas_lc["flux"] = flux_vals * flux_sign
    atlas_lc["fluxerr"] = flux_vals / atlas_lc["snr"]
    atlas_lc["flux5sig"] = 10 ** (-0.4 * (atlas_lc["mag5sig"] - 23.9))

    atlas_lc.sort_values("mjd", inplace=True)

    if average_epochs:
        # Array trick to group observations with are "close" (<rolling_window)
        mjd_group = (
            atlas_lc["mjd"] > atlas_lc["mjd"].shift() + rolling_window
        ).cumsum()
        atlas_lc["mjd_group"] = mjd_group

        # For each group of observations, take the mean flux, weighted err.
        row_list = []
        for (mjd_id, f_id), group in atlas_lc.groupby(["mjd_group", "F"]):
            # Drop some string columns so we can average column-wise.
            group.drop(["Obs", "F"], inplace=True, axis=1)

            row = group.mean()
            row["F"] = f_id  # Have to re-add, we needed to drop it so the ave. worked.
            row["N_exp"] = len(group)
            if len(group) == 1:
                row_list.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                weights = 1.0 / (group["fluxerr"] ** 2)
                flux = np.sum(weights * group["flux"]) / np.sum(weights)
                fluxerr = 1.0 / np.sqrt(np.sum(weights))
                row["flux"] = flux
                row["fluxerr"] = fluxerr
                row["mag5sig"] = (-2.5 * np.log10(row["flux5sig"])) + 23.9
            row_list.append(row)

        atlas_lc = pd.DataFrame(row_list)
        atlas_lc["snr"] = atlas_lc["flux"] / atlas_lc["fluxerr"]
        flux_sign = np.sign(atlas_lc["flux"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            atlas_lc["m"] = (-2.5 * np.log10(abs(atlas_lc["flux"])) + 23.9) * flux_sign
        atlas_lc["dm"] = dm_snr / abs(atlas_lc["snr"])
    else:
        atlas_lc["N_exp"] = 1

    atlas_lc.sort_values("mjd", inplace=True, ignore_index=True)
    atlas_lc.reset_index(drop=True, inplace=True)

    atlas_lc["source"] = "atlas"

    # Set upper lim tag here - if mag is fainter than lim, or mag is negative.
    tag_data = np.full(len(atlas_lc), valid_tag, dtype="object")  # Default to valid.
    upperlim_mask = (atlas_lc["m"] > atlas_lc["mag5sig"]) | (atlas_lc["m"] < 0)
    tag_data[upperlim_mask] = ulimit_tag
    atlas_lc.loc[:, "tag"] = pd.Series(tag_data)

    # Fix other columns here - remap band columns (for eg. sncosmo)
    atlas_band_lookup = {"o": "atlaso", "c": "atlasc"}
    atlas_lc.loc[:, "band"] = atlas_lc["F"].map(atlas_band_lookup)
    # with pd.option_context("mode.chained_assignment", None):
    #     atlas_lc.loc[:, "jd"] = Time(atlas_lc["mjd"].values, format="mjd").jd

    # Fix column names here
    atlas_colmap = {"m": "mag", "dm": "magerr", "mag5sig": "diffmaglim"}
    atlas_lc.rename(atlas_colmap, axis=1, inplace=True)
    use_cols = "mjd mag magerr diffmaglim tag band N_exp source".split()
    return atlas_lc[use_cols]


def prepare_yse_data(
    yse_data: TargetData,
    use_all_sources: bool = True,
    additional_sources: tuple[str] = (),
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):

    if isinstance(additional_sources, str):
        additional_sources = [additional_sources]

    ps1_lookup = {band: f"ps1::{band}" for band in "w g r i z y".split()}
    swift_lookup = {
        band.upper(): f"uvot::{band}" for band in "b u uvm1 uvm2 uvw1 uvw2 v".split()
    }
    atlas_lookup = {"orange-ATLAS": "atlaso", "cyan-ATLAS": "atlasc"}
    ztf_lookup = {"g-ZTF": "ztfg", "r-ZTF": "ztfr", "i-ZTF": "ztfi"}
    unknown = {"Unknown": "unknown", "unknown": "unknown"}
    yse_band_lookup = {
        **ps1_lookup,
        **swift_lookup,
        **atlas_lookup,
        **ztf_lookup,
        **unknown,
    }

    source_lookup = {
        "P48": "ztf",
        "Pan-STARRS1": "yse",
        "Pan-STARRS2": "yse",
        "Swift": "swift",
        "Unknown": "unknown",
    }

    yse_lc = yse_data.lightcurve.copy()

    bad_dq_mask = yse_lc["dq"].astype("str").str.lower() == "bad"
    yse_lc = yse_lc[~bad_dq_mask]

    if "tag" not in yse_lc:
        yse_lc["tag"] = valid_tag
    yse_lc["band"] = yse_lc["flt"].map(yse_band_lookup)
    yse_lc["source"] = yse_lc["instrument"].map(source_lookup)

    if not use_all_sources:
        useful_mask = yse_lc["source"] == "yse"
        for source in additional_sources:
            source_mask = yse_lc["source"].str.lower() == source.lower()
            useful_mask = useful_mask | source_mask
        yse_lc = yse_lc[useful_mask]

    # yse_lc["tag"] = valid_tag
    # TODO: properly set tag for limits, etc.

    use_cols = "mjd mag magerr band tag source".split()

    return yse_lc[use_cols]


class DefaultLightcurveCompiler:
    __name__ = "default_lightcurve_compiler"

    valid_tag = DEFAULT_VALID_TAG
    badqual_tag = DEFAULT_BADQUAL_TAG
    ulimit_tag = DEFAULT_ULIMIT_TAG
    default_broker_priority = ("fink", "alerce", "lasair", "ampel", "antares")

    def __init__(
        self,
        atlas_average_epochs: bool = True,
        atlas_rolling_window: float = 0.1,
        broker_priority: list[str] = None,
        yse_use_all_sources: bool = True,
        yse_additional_sources: tuple[str] = (),
    ):
        self.atlas_average_epochs = atlas_average_epochs
        self.atlas_rolling_window = atlas_rolling_window
        self.broker_priority = broker_priority or self.default_broker_priority
        self.yse_use_all_sources = yse_use_all_sources
        self.yse_additional_sources = yse_additional_sources

    def __call__(self, target: Target, t_ref: Time):
        lightcurve_dfs = []

        tags = {
            "valid_tag": self.valid_tag,
            "ulimit_tag": self.ulimit_tag,
            "badqual_tag": self.badqual_tag,
        }

        target_id = target.target_id

        ##===== Prepare ZTF data =====##
        # Select the best data from the ZTF brokers
        ztf_data = target.target_data.get("ztf", None)  # Probably never happens...
        if ztf_data is None:
            for broker in self.broker_priority:
                source_name = f"{broker}_ztf"
                broker_data = target.target_data.get(source_name, None)
                if broker_data is None:
                    continue  # try next favouite broker...
                if not _check_is_target_data(
                    broker_data, source_name=source_name, target_id=target_id
                ):
                    continue
                if broker_data.lightcurve is not None:
                    ztf_data = broker_data
                    break

        # If it exists, format it nicely.
        if ztf_data is not None:
            try:
                ztf_lc = prepare_ztf_data(ztf_data, **tags)
                lightcurve_dfs.append(ztf_lc)
            except Exception as e:
                logger.error(e)
                msg = f"can't process ztf source {broker}: {target.target_id}"
                raise ValueError(msg)

        ##===== Prepare the LSST data =====##
        # Get best data from LSST brokers
        lsst_data = target.target_data.get("lsst", None)  # Probably never happens...
        if lsst_data is None:
            for broker in self.broker_priority:
                source_name = f"{broker}_lsst"
                broker_data = target.target_data.get(source_name, None)
                if broker_data is None:
                    continue  # try next favourite broker...
                if not _check_is_target_data(
                    broker_data, source_name=source_name, target_id=target_id
                ):
                    continue
                if broker_data.lightcurve is not None:
                    lsst_data = broker_data

        # If it exists, format it nicely (to match everything else)
        if lsst_data is not None:
            try:
                lsst_lc = prepare_lsst_data(lsst_data, **tags)
                lightcurve_dfs.append(lsst_lc)
            except Exception as e:
                logger.error(e)
                msg = f"can't process lsst source {broker}"

        ##===== Prepare the ATLAS data =====##
        atlas_data = target.target_data.get("atlas", None)
        if atlas_data is not None:
            atlas_lc = atlas_data.lightcurve
            if (atlas_lc is not None) and (not atlas_lc.empty):
                atlas_df = prepare_atlas_data(
                    atlas_data,
                    average_epochs=self.atlas_average_epochs,
                    rolling_window=self.atlas_rolling_window,
                    **tags,
                )
                if not (len(atlas_df) == 0 or atlas_df.empty):
                    lightcurve_dfs.append(atlas_df)

        ##===== Prepare the YSE data =====##
        yse_data = target.target_data.get("yse", None)
        if yse_data is not None:
            yse_lc = yse_data.lightcurve
            if (yse_lc is not None) and (not yse_lc.empty):
                yse_df = prepare_yse_data(
                    yse_data,
                    use_all_sources=self.yse_use_all_sources,
                    additional_sources=self.yse_use_all_sources,
                    **tags,
                )
                if not (len(yse_df) == 0 or yse_df.empty):
                    lightcurve_dfs.append(yse_df)

        ##===== Stitch it all together =====##
        compiled_lightcurve = None
        if len(lightcurve_dfs) > 0:
            compiled_lightcurve = pd.concat(lightcurve_dfs, ignore_index=True)
            compiled_lightcurve.sort_values("mjd", inplace=True, ignore_index=True)

            if any(pd.isna(compiled_lightcurve["band"])):
                logger.warning(f"band 'NaN' in {target_id} compiled_lightcurve")
        return compiled_lightcurve
