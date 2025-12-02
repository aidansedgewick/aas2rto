import numpy as np

import pandas as pd

from astropy.time import Time

from astroplan import Observer

from aas2rto.target import Target

def constant_score(target: Target, t_ref: Time):
    return 1.0


def base_score(target: Target, t_ref: Time):
    return target.base_score, ["just return base_score"]


def latest_flux(target: Target, t_ref: Time) -> float:
    ztf_data = None

    score_comments = []

    exclude = False
    reject = False

    broker_priority = ("ztf", "alerce")

    for broker in broker_priority:
        source_name = f"ztf_{broker}"
        source_data = target.target_data.get(source_name)
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
            f"no ztf data from {broker_priority}. only {target.target_data.keys()}"
        )
        return -1.0, score_comments
    else:
        latest_mag = ztf_data.detections["magpsf"].iloc[-1]

    latest_flux = 3631.0 * 10 ** (-0.4 * latest_mag)  # in Jy
    score = latest_flux * 10**6  # in uJy
    score_comments.append(f"latest_mag {latest_mag:.2f}")

    if exclude:
        score = -1.0

    return score, score_comments
