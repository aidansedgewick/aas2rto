import copy
from logging import getLogger

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets.target import DEFAULT_ZTF_BROKER_PRIORITY
from dk154_targets import Target, TargetData

# from dk154_targets.query_managers import atlas

logger = getLogger(__name__.split(".")[-1])


def prepare_ztf_data(ztf_data: pd.DataFrame):
    ztf_band_lookup = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    col_map = {"magpsf": "mag", "sigmapsf": "magerr", "fid": "band"}
    use_cols = ["mjd", "jd", "magpsf", "sigmapsf", "diffmaglim", "tag"]

    avail_cols = [col for col in use_cols if col in ztf_data.columns]
    with pd.option_context("mode.chained_assignment", None):
        ztf_df = ztf_data[avail_cols]
        ztf_df.loc[:, "band"] = ztf_data["fid"].map(ztf_band_lookup)
        ztf_df.rename(col_map, axis=1, inplace=True)
    return ztf_df


def prepare_atlas_data(atlas_data: pd.DataFrame, average_epochs=True):
    atlas_band_lookup = {"o": "atlaso", "c": "atlasc"}

    atlas_cols = ["m", "dm", "mag5sig"]
    atlas_rename = {"m": "mag", "dm": "magerr", "mag5sig": "diffmaglim"}

    atlas_df = copy.deepcopy(atlas_data)

    # atlas_df["snr"] = atlas_df["uJy"] / atlas_df["duJy"]
    atlas_df["snr"] = 2.5 / (np.log(10.0) * atlas_df["dm"])

    flux_vals = 10 ** (-0.4 * (abs(atlas_df["m"]) - 23.9))
    flux_sign = flux_vals / abs(flux_vals)
    atlas_df["flux"] = flux_vals * flux_sign
    atlas_df["fluxerr"] = atlas_df["flux"] / atlas_df["snr"]

    atlas_df.sort_values("mjd", inplace=True)

    if average_epochs:
        atlas_df.sort_values("mjd")
        mjd_group = (atlas_df["mjd"] > atlas_df["mjd"].shift() + 0.1).cumsum()
        atlas_df["mjd_group"] = mjd_group

        row_list = []
        for (mjd_id, f_id), group in atlas_df.groupby(["mjd_group", "F"]):
            group.drop(["Obs", "F"], inplace=True, axis=1)

            row = group.mean()

            row["F"] = f_id
            if len(group) == 1:
                row_list.append(row)

            weights = 1.0 / (group["fluxerr"] ** 2)
            flux = np.sum(weights * group["flux"]) / np.sum(weights)
            fluxerr = 1.0 / np.sqrt(np.sum(weights))
            row["flux"] = flux
            row["fluxerr"] = fluxerr

            row_list.append(row)

        atlas_df = pd.DataFrame(row_list)

        atlas_df["snr"] = atlas_df["flux"] / atlas_df["fluxerr"]

        flux_sign = np.sign(atlas_df["flux"])
        atlas_df["m"] = (-2.5 * np.log10(abs(atlas_df["flux"])) + 23.9) * flux_sign
        atlas_df["dm"] = (np.log(10) / 2.5) / abs(atlas_df["snr"])

    atlas_df.reset_index(drop=True, inplace=True)

    tag_data = np.full(len(atlas_df), "valid", dtype="object")
    upperlim_mask = (atlas_df["m"] > atlas_df["mag5sig"]) | (atlas_df["m"] < 0)
    tag_data[upperlim_mask] = "upperlim"

    with pd.option_context("mode.chained_assignment", None):
        atlas_df.loc[:, "band"] = atlas_df["F"].map(atlas_band_lookup)
        atlas_df.loc[:, "jd"] = Time(atlas_df["mjd"].values, format="mjd").jd
        atlas_df.loc[:, "tag"] = pd.Series(tag_data)

    atlas_df.rename(atlas_rename, axis=1, inplace=True)
    use_cols = ["mjd", "jd", "mag", "magerr", "diffmaglim", "tag", "band"]

    return atlas_df[use_cols]


def prepare_yse_data(yse_data: pd.DataFrame):
    yse_band_lookup = {band: f"ps1::{band}" for band in "g r i z y".split()}

    yse_df = yse_data.copy()
    yse_df["band"] = yse_df["flt"].map(yse_band_lookup)

    return yse_df


class DefaultLightcurveCompiler:
    def __init__(self, average_atlas_epochs=True, **config):
        self.average_atlas_epochs = average_atlas_epochs

        for key, val in config.items():
            logger.warning(f"unknown config option: {key} ({val})")

    def __call__(self, target: Target):
        lightcurve_dfs = []

        # Select the best data from the brokers.
        broker_data = None
        for source in DEFAULT_ZTF_BROKER_PRIORITY:
            source_data = getattr(target, f"{source}_data", None)
            if not isinstance(source_data, TargetData):
                continue
            if source_data.lightcurve is not None:
                broker_data = source_data.lightcurve
                break

        if broker_data is not None:
            if "mjd" in broker_data.columns and "jd" not in broker_data.columns:
                broker_data["jd"] = Time(broker_data["mjd"].values, format="mjd").jd

            try:
                broker_data = prepare_ztf_data(broker_data)
                lightcurve_dfs.append(broker_data)
            except Exception as e:
                print(e)
                raise ValueError(f"can't process df:\n{broker_data}")

        # Get ATLAS data
        if target.atlas_data.lightcurve is not None:
            if not target.atlas_data.lightcurve.empty:
                atlas_df = prepare_atlas_data(
                    target.atlas_data.lightcurve,
                    average_epochs=self.average_atlas_epochs,
                )
                lightcurve_dfs.append(atlas_df)

        # Get YSE data
        if target.yse_data.lightcurve is not None:
            if not target.yse_data.lightcurve.empty:
                yse_df = prepare_yse_data(target.yse_data.lightcurve)
                lightcurve_dfs.append(yse_df)

        compiled_lightcurve = None
        if len(lightcurve_dfs) > 0:
            compiled_lightcurve = pd.concat(lightcurve_dfs, ignore_index=True)
            compiled_lightcurve.sort_values("jd", inplace=True)
        return compiled_lightcurve


def default_compile_lightcurve(target: Target):
    # TODO: is this better as a class with a __call__ magic method?
    lightcurve_dfs = []

    # Select the best data from the brokers.
    broker_data = None
    for source in DEFAULT_ZTF_BROKER_PRIORITY:
        source_data = getattr(target, f"{source}_data", None)
        if not isinstance(source_data, TargetData):
            continue
        if source_data.lightcurve is not None:
            broker_data = source_data.lightcurve
            break

    if broker_data is not None:
        if "mjd" in broker_data.columns and "jd" not in broker_data.columns:
            broker_data["jd"] = Time(broker_data["mjd"].values, format="mjd").jd

        try:
            broker_data = prepare_ztf_data(broker_data)
            lightcurve_dfs.append(broker_data)
        except Exception as e:
            print(e)
            raise ValueError(f"can't process df:\n{broker_data}")

    # Get the atlas data
    if target.atlas_data.lightcurve is not None:
        if not target.atlas_data.lightcurve.empty:
            atlas_df = prepare_atlas_data(target.atlas_data.lightcurve)
            lightcurve_dfs.append(atlas_df)

    # Get YSE data
    if target.yse_data.lightcurve is not None:
        if not target.yse_data.lightcurve.empty:
            yse_df = prepare_yse_data(target.yse_data.lightcurve)
            lightcurve_dfs.append(yse_df)

    compiled_lightcurve = None
    if len(lightcurve_dfs) > 0:
        compiled_lightcurve = pd.concat(lightcurve_dfs, ignore_index=True)
        compiled_lightcurve.sort_values("jd", inplace=True)
    return compiled_lightcurve
