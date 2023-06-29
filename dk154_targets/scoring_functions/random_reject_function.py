import numpy as np

from astroplan import Observer

from dk154_targets import Target

def random_reject(target: Target, observatory: Observer):
    return -np.inf if np.random.uniform() > 0.5 else 100.