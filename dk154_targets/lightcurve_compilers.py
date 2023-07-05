import copy

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets import Target

# from dk154_targets.query_managers import atlas


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


def prepare_atlas_data(atlas_data: pd.DataFrame):
    atlas_band_lookup = {"o": "atlaso", "c": "atlasc"}

    atlas_cols = ["m", "dm", "mag5sig"]
    atlas_rename = {"m": "mag", "dm": "magerr", "mag5sig": "diffmaglim"}

    atlas_df = copy.deepcopy(atlas_data)
    atlas_df.query("m > 0", inplace=True)
    atlas_df.reset_index(drop=True, inplace=True)

    tag_data = np.full(len(atlas_df), "valid", dtype="object")
    upperlim_mask = atlas_df["m"] > atlas_df["mag5sig"]
    tag_data[upperlim_mask] = "upperlim"

    with pd.option_context("mode.chained_assignment", None):
        atlas_df.loc[:, "band"] = atlas_df["F"].map(atlas_band_lookup)
        atlas_df.loc[:, "jd"] = Time(atlas_df["mjd"].values, format="mjd").jd
        atlas_df.loc[:, "tag"] = pd.Series(tag_data)

    atlas_df.rename(atlas_rename, axis=1, inplace=True)
    use_cols = ["mjd", "jd", "mag", "magerr", "diffmaglim", "tag", "band"]

    return atlas_df[use_cols]


def default_compile_lightcurve(target: Target):
    # TODO: is this better as a class with a __call__ magic method?
    lightcurve_dfs = []

    # Select the best data from the brokers.
    source_data_list = [
        getattr(target, f"{broker}_data", None) for broker in target.broker_priority
    ]
    if not all(source_data_list):  # TargetData evaluates as True.
        raise ValueError("some source data is None! (Should be TargetData")
    broker_data = None
    for source_data in source_data_list:
        if source_data.lightcurve is not None:
            broker_data = source_data.lightcurve
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

    compiled_lightcurve = None
    if len(lightcurve_dfs) > 0:
        compiled_lightcurve = pd.concat(lightcurve_dfs, ignore_index=True)
        compiled_lightcurve.sort_values("jd", inplace=True)
    return compiled_lightcurve
