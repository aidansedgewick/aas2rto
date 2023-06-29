import numpy as np

import pandas as pd

from astroplan import Observer

from dk154_targets import Target

def random_reject(target: Target, observatory: Observer):
    return -np.inf if np.random.uniform() > 0.5 else 100.

def peak_flux(target: Target, observatory: Observer) -> float:
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
        return -np.inf

    peak_mag = ztf_data.detections.groupby("fid")["magpsf"].min().min()
    peak_flux = 3631 * 10 ** (-0.4 * peak_mag)  # in Jy
    return peak_flux * 10**9  # in nJy
