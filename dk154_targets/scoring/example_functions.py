import numpy as np

import pandas as pd

from astropy.time import Time

from astroplan import Observer

from dk154_targets import Target


def empty_scoring(target: Target, obs: Observer, t_ref: Time):
    return 100.0


def base_score(target: Target, obs: Observer, t_ref: Time):
    return target.base_score


def random_reject(target: Target, observatory: Observer, t_ref: Time):
    return -np.inf if np.random.uniform() > 0.5 else target.base_score


def peak_flux(target: Target, observatory: Observer, t_ref: Time) -> float:
    ztf_priority = ("alerce", "fink", "lasair")

    ztf_data = None
    for broker in ztf_priority:
        data_name = f"{broker}_data"
        source_data = getattr(target, data_name, None)
        if source_data is None:
            continue
        if source_data.lightcurve is None:
            continue
        ztf_data = source_data
        break
    if ztf_data is None:
        return -1.0

    peak_mag = ztf_data.detections.groupby("fid")["magpsf"].min().min()
    peak_flux = 3631 * 10 ** (-0.4 * peak_mag)  # in Jy
    return peak_flux * 10**9  # in nJy


def latest_flux(target: Target, observatory: Observer, t_ref: Time) -> float:
    ztf_priority = ("alerce", "fink", "lasair")
    ztf_data = None
    for broker in ztf_priority:
        data_name = f"{broker}_data"
        source_data = getattr(target, data_name, None)
        if source_data is None:
            continue
        if source_data.lightcurve is None:
            continue
        ztf_data = source_data
        break
    if ztf_data is None:
        return -1

    latest_mag = ztf_data.detections["magpsf"].iloc[-1]
    latest_flux = 3631 * 10 ** (-0.4 * latest_mag)  # in Jy
    return latest_flux * 10**9  # in nJy
