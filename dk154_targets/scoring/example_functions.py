import numpy as np

import pandas as pd

from astropy.time import Time

from astroplan import Observer

from dk154_targets.target import Target, DEFAULT_ZTF_BROKER_PRIORITY


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
    ztf_data = None

    score_comments = []
    reject_comments = []

    exclude = False
    reject = False

    for broker in DEFAULT_ZTF_BROKER_PRIORITY:
        source_data = target.target_data.get(broker)
        if source_data is None:
            continue
        if source_data.lightcurve is None:
            continue
        ztf_data = source_data
        score_comments.append(f"ztf data from {broker}")
        break

    if ztf_data is None:
        exclude = True
        score_comments.append(
            f"no ztf data from {DEFAULT_ZTF_BROKER_PRIORITY}. only {target.target_data.keys()}"
        )
        latest_mag = 99.0
    else:
        latest_mag = ztf_data.detections["magpsf"].iloc[-1]

    latest_flux = 3631.0 * 10 ** (-0.4 * latest_mag)  # in Jy
    score = latest_flux * 10**9  # in nJy
    score_comments.append(f"latest_mag {latest_mag:.2f}")

    if exclude:
        score = -1.0

    return score, score_comments, reject_comments


def latest_flux_atlas_requirement(
    target: Target, observatory: Observer, t_ref: Time
) -> float:

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

    if target.atlas_data.lightcurve is None:
        return -1

    latest_mag = ztf_data.detections["magpsf"].iloc[-1]
    latest_flux = 3631 * 10 ** (-0.4 * latest_mag)  # in Jy
    return latest_flux * 10**9  # in nJy
