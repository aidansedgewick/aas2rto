import copy
import time
from logging import getLogger

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto.target import DEFAULT_ZTF_BROKER_PRIORITY
from aas2rto import Target, TargetData

# from dk154_targets.query_managers import atlas

logger = getLogger(__name__.split(".")[-1])

DEFAULT_VALID_TAG = "valid"
DEFAULT_BADQUAL_TAG = "badqual"
DEFAULT_ULIMIT_TAG = "upperlim"


def prepare_ztf_data(
    ztf_data: TargetData,
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):
    # Rename the filters so sncosmo knows what they are.
    ztf_band_lookup = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    ztf_colmap = {"magpsf": "mag", "sigmapsf": "magerr"}
    use_cols = "jd mjd magpsf sigmapsf diffmaglim fid tag candid".split()

    if isinstance(ztf_data, TargetData):

        avail_cols = [col for col in use_cols if col in ztf_data.lightcurve.columns]

        data_list = []
        if ztf_data.detections is not None:
            detections = ztf_data.detections[avail_cols].copy()
            detections.loc[:, "tag"] = valid_tag
            # data_list = [detections]
            data_list.append(detections)
        if ztf_data.badqual is not None:
            badqual = ztf_data.badqual[avail_cols].copy()
            badqual.loc[:, "tag"] = badqual_tag
            data_list.append(badqual)
        if ztf_data.non_detections is not None:
            ulimits = ztf_data.non_detections[avail_cols].copy()
            ulimits.loc[:, "tag"] = ulimit_tag
            data_list.append(ulimits)
        if len(data_list) > 0:
            ztf_lc = pd.concat(data_list, ignore_index=True)
        else:
            ztf_lc = ztf_data.lightcurve

        if len(detections) == 0:
            print(ztf_data.lightcurve[avail_cols])
    else:
        logger.warning(f"")
        ztf_lc = ztf_data

    ztf_lc["source"] = "ztf"

    # fix column names here
    ztf_lc.loc[:, "band"] = ztf_lc["fid"].map(ztf_band_lookup)
    ztf_lc.sort_values("jd", inplace=True)
    ztf_lc.rename(ztf_colmap, axis=1, inplace=True)
    if "mjd" not in ztf_lc.columns:
        ztf_lc.insert(1, "mjd", Time(ztf_lc["jd"], format="jd").mjd)

    return ztf_lc


def prepare_atlas_data(
    atlas_data: TargetData,
    average_epochs=True,
    rolling_window=0.1,
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):
    atlas_band_lookup = {"o": "atlaso", "c": "atlasc"}
    # atlas_cols = ["m", "dm", "mag5sig"]
    atlas_colmap = {"m": "mag", "dm": "magerr", "mag5sig": "diffmaglim"}

    atlas_lc = atlas_data.lightcurve.copy()

    # atlas_df["snr"] = atlas_df["uJy"] / atlas_df["duJy"]

    dm_snr = 2.5 / np.log(10.0)
    atlas_lc["snr"] = dm_snr / atlas_lc["dm"]

    flux_vals = 10 ** (-0.4 * (abs(atlas_lc["m"]) - 23.9))
    flux_sign = np.sign(atlas_lc["m"])
    atlas_lc["flux"] = flux_vals * flux_sign
    atlas_lc["fluxerr"] = flux_vals / atlas_lc["snr"]
    atlas_lc["flux5sig"] = 10 ** (-0.4 * (atlas_lc["mag5sig"] - 23.9))

    atlas_lc.sort_values("mjd", inplace=True)

    if average_epochs:
        mjd_group = (
            atlas_lc["mjd"] > atlas_lc["mjd"].shift() + rolling_window
        ).cumsum()
        atlas_lc["mjd_group"] = mjd_group

        row_list = []
        for (mjd_id, f_id), group in atlas_lc.groupby(["mjd_group", "F"]):
            group.drop(["Obs", "F"], inplace=True, axis=1)  # so can compute mean on it.

            row = group.mean()
            row["F"] = f_id
            row["N_exp"] = len(group)
            if len(group) == 1:
                row_list.append(row)
                continue

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
        atlas_lc["m"] = (-2.5 * np.log10(abs(atlas_lc["flux"])) + 23.9) * flux_sign
        atlas_lc["dm"] = dm_snr / abs(atlas_lc["snr"])
    else:
        atlas_lc["N_exp"] = 1

    atlas_lc.reset_index(drop=True, inplace=True)

    tag_data = np.full(len(atlas_lc), valid_tag, dtype="object")
    upperlim_mask = (atlas_lc["m"] > atlas_lc["mag5sig"]) | (atlas_lc["m"] < 0)
    tag_data[upperlim_mask] = ulimit_tag

    with pd.option_context("mode.chained_assignment", None):
        atlas_lc.loc[:, "band"] = atlas_lc["F"].map(atlas_band_lookup)
        atlas_lc.loc[:, "jd"] = Time(atlas_lc["mjd"].values, format="mjd").jd
        atlas_lc.loc[:, "tag"] = pd.Series(tag_data)

    atlas_lc["source"] = "atlas"

    atlas_lc.rename(atlas_colmap, axis=1, inplace=True)
    use_cols = [
        "mjd",
        "jd",
        "mag",
        "magerr",
        "diffmaglim",
        "tag",
        "band",
        "N_exp",
        "source",
    ]
    return atlas_lc[use_cols]


def prepare_yse_data(
    yse_data: TargetData,
    valid_tag=DEFAULT_VALID_TAG,
    badqual_tag=DEFAULT_BADQUAL_TAG,
    ulimit_tag=DEFAULT_ULIMIT_TAG,
):
    ps1_lookup = {band: f"ps1::{band}" for band in "w g r i z y".split()}
    swift_lookup = {band: f"uvot::{band}" for band in "b u uvm1 uvw1 uvw2 v".split()}
    atlas_lookup = {"orange-ATLAS": "atlaso", "cyan-ATLAS": "atlasc"}
    ztf_lookup = {"g-ZTF": "ztfg", "r-ZTF": "ztfr", "i-ZTF": "ztfi"}
    yse_band_lookup = {**ps1_lookup, **swift_lookup, **atlas_lookup, **ztf_lookup}

    source_lookup = {
        "P48": "ztf",
        "Pan-STARRS1": "yse",
        "Pan-STARRS2": "yse",
        "Swift": "swift",
    }

    yse_lc = yse_data.lightcurve.copy()
    yse_lc["band"] = yse_lc["flt"].map(yse_band_lookup)
    yse_lc["source"] = yse_lc["instrument"].map(source_lookup)

    # yse_lc["tag"] = valid_tag
    # TODO: properly set tag for limits, etc.

    use_cols = ["mjd", "jd", "mag", "magerr", "tag", "band", "source"]

    return yse_lc[use_cols]


class DefaultLightcurveCompiler:
    __name__ = "default_lightcurve_compiler"

    valid_tag = DEFAULT_VALID_TAG
    badqual_tag = DEFAULT_BADQUAL_TAG
    ulimit_tag = DEFAULT_ULIMIT_TAG

    def __init__(self, average_atlas_epochs=True, ztf_broker_priority=None, **config):
        self.average_atlas_epochs = average_atlas_epochs
        self.ztf_broker_priority = ztf_broker_priority or DEFAULT_ZTF_BROKER_PRIORITY

        for key, val in config.items():
            logger.warning(f"unknown config option: {key} ({val})")

    def __call__(self, target: Target, t_ref: Time):
        lightcurve_dfs = []

        tags = {
            "valid_tag": self.valid_tag,
            "ulimit_tag": self.ulimit_tag,
            "badqual_tag": self.badqual_tag,
        }

        target_id = target.target_id

        # Select the best data from the ZTF brokers.
        broker_data = None
        for ztf_source in self.ztf_broker_priority:
            source_data = target.target_data.get(ztf_source)
            if source_data is None:
                continue
            if not isinstance(source_data, TargetData):
                continue
            if source_data.lightcurve is not None:
                broker_data = source_data
                break

        if broker_data is not None:
            try:
                ztf_lc = prepare_ztf_data(broker_data, **tags)
                lightcurve_dfs.append(ztf_lc)
            except Exception as e:
                print(e)
                msg = f"can't process ztf_source {ztf_source}: {target.target_id}"
                raise ValueError(msg)

        # Get ATLAS data
        atlas_data = target.target_data.get("atlas", None)
        if atlas_data is not None:
            atlas_lc = atlas_data.lightcurve
            if (atlas_lc is not None) and (not atlas_lc.empty):
                atlas_df = prepare_atlas_data(
                    atlas_data, average_epochs=self.average_atlas_epochs, **tags
                )
                if not (len(atlas_df) == 0 or atlas_df.empty):
                    lightcurve_dfs.append(atlas_df)

        # Get YSE data
        yse_data = target.target_data.get("yse", None)
        if yse_data is not None:
            yse_lc = yse_data.lightcurve
            if (yse_lc is not None) and (not yse_lc.empty):
                yse_df = prepare_yse_data(yse_data)
                if not (len(yse_df) == 0 or yse_df.empty):
                    lightcurve_dfs.append(yse_df)
                yse_df["source"] = "yse"

        compiled_lightcurve = None
        if len(lightcurve_dfs) > 0:
            compiled_lightcurve = pd.concat(lightcurve_dfs, ignore_index=True)
            compiled_lightcurve.sort_values("mjd", inplace=True, ignore_index=True)
        return compiled_lightcurve
