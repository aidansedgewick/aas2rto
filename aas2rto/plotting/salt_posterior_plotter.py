from pathlib import Path

import numpy as np

from astropy.time import Time

from aas2rto.target import Target


def plot_salt_posterior(target, t_ref: Time = None):
    pass


class SaltPoseteriorPlotter:

    plotting_directory = "post"
    default_figsize = ()

    @classmethod
    def plot(cls, target: Target, **kwargs):
        plotter = cls(**kwargs)
        plotter.plot_target(target)

    def __init__(self, figsize: tuple = None, **kwargs):

        self.init_fig()

    def set_default_plot_params(self):
        pass

    def init_fig(figsize=figsize):
        pass
