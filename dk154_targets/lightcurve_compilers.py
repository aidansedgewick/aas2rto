import copy
from logging import getLogger

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets.target import DEFAULT_ZTF_BROKER_PRIORITY
from dk154_targets import Target, TargetData

# from dk154_targets.query_managers import atlas

logger = getLogger(__name__.split(".")[-1])


def prepare_ztf_data(
    ztf_data: TargetData,
    valid_tag="valid",
    badqual_tag="badqual",
    ulimit_tag="upperlim",
):
    ztf_band_lookup = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    ztf_colmap = {"magpsf": "mag", "sigmapsf": "magerr", "fid": "band"}
    use_cols = ["jd", "magpsf", "sigmapsf", "diffmaglim", "fid", "tag"]

    detections = ztf_data.detections.copy()
    detections.loc[:, "tag"] = valid_tag
    bad_qual = ztf_data.badqual.copy()
    bad_qual.loc[:, "tag"] = badqual_tag
    ulimits = ztf_data.non_detections.copy()
    ulimits.loc[:, "tag"] = ulimit_tag
    ztf_lc = pd.concat([detections, bad_qual, ulimits], ignore_index=True)

    ztf_lc.loc[:, "band"] = ztf_lc["fid"].map(ztf_band_lookup)
    avail_cols = [col for col in use_cols if col in ztf_lc.columns]
    missing_cols = [col for col in use_cols if col not in ztf_lc.columns]
    if len(missing_cols) > 0:
        logger.warning(f"columns unavailable: {missing_cols}")

    ztf_lc = ztf_lc[avail_cols]
    ztf_lc.sort_values("jd", inplace=True)
    ztf_lc.rename(ztf_colmap, axis=1, inplace=True)
    if "mjd" not in ztf_lc.columns:
        ztf_lc.insert(1, "mjd", Time(ztf_lc["jd"], format="jd").mjd)
    return ztf_lc


# def prepare_ztf_lc():
#    pass


def prepare_atlas_data(
    atlas_data: TargetData,
    average_epochs=True,
    valid_tag="valid",
    badqual_tag="badqual",
    ulimit_tag="upperlim",
):
    atlas_band_lookup = {"o": "atlaso", "c": "atlasc"}
    # atlas_cols = ["m", "dm", "mag5sig"]
    atlas_rename = {"m": "mag", "dm": "magerr", "mag5sig": "diffmaglim"}

    atlas_lc = atlas_data.lightcurve.copy()

    # atlas_df["snr"] = atlas_df["uJy"] / atlas_df["duJy"]
    atlas_lc["snr"] = 2.5 / (np.log(10.0) * atlas_lc["dm"])

    flux_vals = 10 ** (-0.4 * (abs(atlas_df["m"]) - 23.9))
    flux_sign = flux_vals / abs(flux_vals)
    atlas_lc["flux"] = flux_vals * flux_sign
    atlas_lc["fluxerr"] = atlas_lc["flux"] / atlas_lc["snr"]

    atlas_lc.sort_values("mjd", inplace=True)

    if average_epochs:
        mjd_group = (atlas_lc["mjd"] > atlas_lc["mjd"].shift() + 0.1).cumsum()
        atlas_lc["mjd_group"] = mjd_group

        row_list = []
        for (mjd_id, f_id), group in atlas_lc.groupby(["mjd_group", "F"]):
            group.drop(["Obs", "F"], inplace=True, axis=1)  # so can compute mean on it.

            row = group.mean()
            row["F"] = f_id
            if len(group) == 1:
                row_list.append(row)
                continue

            weights = 1.0 / (group["fluxerr"] ** 2)
            flux = np.sum(weights * group["flux"]) / np.sum(weights)
            fluxerr = 1.0 / np.sqrt(np.sum(weights))
            row["flux"] = flux
            row["fluxerr"] = fluxerr
            row_list.append(row)

        atlas_lc = pd.DataFrame(row_list)
        atlas_lc["snr"] = atlas_lc["flux"] / atlas_lc["fluxerr"]
        flux_sign = np.sign(atlas_lc["flux"])
        atlas_lc["m"] = (-2.5 * np.log10(abs(atlas_lc["flux"])) + 23.9) * flux_sign
        atlas_lc["dm"] = (np.log(10) / 2.5) / abs(atlas_lc["snr"])

    atlas_lc.reset_index(drop=True, inplace=True)

    tag_data = np.full(len(atlas_lc), valid_tag, dtype="object")
    upperlim_mask = (atlas_lc["m"] > atlas_lc["mag5sig"]) | (atlas_lc["m"] < 0)
    tag_data[upperlim_mask] = ulimit_tag

    with pd.option_context("mode.chained_assignment", None):
        atlas_lc.loc[:, "band"] = atlas_lc["F"].map(atlas_band_lookup)
        atlas_lc.loc[:, "jd"] = Time(atlas_lc["mjd"].values, format="mjd").jd
        atlas_lc.loc[:, "tag"] = pd.Series(tag_data)

    atlas_lc.rename(atlas_rename, axis=1, inplace=True)
    use_cols = ["mjd", "jd", "mag", "magerr", "diffmaglim", "tag", "band"]
    return atlas_lc[use_cols]


def prepare_yse_data(
    yse_data: TargetData,
    valid_tag="valid",
    badqual_tag="badqual",
    ulimit_tag="upperlim",
):
    yse_band_lookup = {band: f"ps1::{band}" for band in "g r i z y".split()}

    yse_lc = yse_data.lightcurve.copy()
    yse_lc["band"] = yse_lc["flt"].map(yse_band_lookup)
    yse_lc["tag"] = vaild_tag
    # TODO: properly set tag for limits, etc.
    return yse_lc


class DefaultLightcurveCompiler:
    valid_tag = "valid"
    ulimit_tag = "upperlim"
    badqual_tag = "badquality"

    def __init__(self, average_atlas_epochs=True, ztf_broker_priority=None, **config):
        self.average_atlas_epochs = average_atlas_epochs
        self.ztf_broker_priority = ztf_broker_priority or DEFAULT_ZTF_BROKER_PRIORITY

        for key, val in config.items():
            logger.warning(f"unknown config option: {key} ({val})")

    def __call__(self, target: Target):
        print(f"compiling {target.objectId}")
        print(f"target_data keys: {target.target_data.keys()}")
        lightcurve_dfs = []

        tags = {
            "valid_tag": self.valid_tag,
            "ulimit_tag": self.ulimit_tag,
            "badqual_tag": self.badqual_tag,
        }

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
                msg = f"can't process ztf_source {ztf_source}: {target.objectId}"
                raise ValueError(msg)

        # Get ATLAS data
        atlas_data = target.target_data.get("atlas", None)
        if atlas_data is not None:
            if (atlas_data.lightcurve) and (not atlas_data.lightcurve.empty):
                atlas_lc = prepare_atlas_data(
                    atlas_data, average_epochs=self.average_atlas_epochs, **tags
                )
                lightcurve_dfs.append(atlas_lc)

        # Get YSE data
        yse_data = target.target_data.get("yse", None)
        if yse_data is not None:
            if (yse_data.lightcurve) and (not yse_data.lightcurve.empty):
                yse_df = prepare_yse_data(
                    yse_data,
                )
                lightcurve_dfs.append(yse_df)

        compiled_lightcurve = None
        if len(lightcurve_dfs) > 0:
            compiled_lightcurve = pd.concat(lightcurve_dfs, ignore_index=True)
            compiled_lightcurve.sort_values("jd", inplace=True, ignore_index=True)
        return compiled_lightcurve
