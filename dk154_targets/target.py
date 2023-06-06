import warnings
from typing import Callable

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroplan import FixedTarget, Observer


class TargetData:
    def __init__(
        self,
        lightcurve: pd.DataFrame = None,
        probabilities: pd.DataFrame = None,
        parameters: dict = None,
        cutouts: dict = None,
        meta: dict = None,
    ):
        self.lightcurve = lightcurve
        self.probabilities = probabilities
        self.parameters = parameters or {}
        self.cutouts = cutouts or {}
        self.meta = dict(
            # lightcurve_update_time=None,
            # probabilities_update_time=None,
            # parameters_update_time=None,
            # cutout_update_time=None,
        )
        meta = meta or {}
        self.meta.update(meta)


class Target:
    """
    TODO: docstring here!
    """

    default_base_score = 100.0
    data_sources = (
        "alerce",
        "atlas",
        "fink",
        "tns",
    )

    def __init__(
        self,
        objectId: str,
        ra: float,
        dec: float,
        alerce_data: TargetData = None,
        atlas_data: TargetData = None,
        fink_data: TargetData = None,
        tns_data: TargetData = None,
        base_score: float = None,
    ):
        # Basics
        self.objectId = objectId
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        self.base_score = base_score or self.default_base_score
        self.astroplan_target = FixedTarget(self.coord, self.objectId)  # for plots

        # Target data
        self.alerce_data = alerce_data or TargetData()
        self.atlas_data = atlas_data or TargetData()
        self.fink_data = fink_data or TargetData()
        self.tns_data = tns_data or TargetData()

        # Observatory data
        self.observatory_night = {"no_observatory": None}

        # Scoring data
        self.models = []
        self.score_history = {"no_observatory": []}
        self.score_comments = {"no_observatory": None}
        self.reject_comments = {"no_observatory": None}
        self.rank_history = {"no_observatory": []}

        # Keep track of what's going on
        self.target_of_opportunity = False
        self.updated = False
        self.to_reject = False

    def evaluate_target(
        self,
        scoring_function: Callable,
        observatory: Observer,
        t_ref: Time = None,
    ):
        """
        Evaluate this Target according to your scoring function.

        Parameters
        ----------
        scoring_function
            a callable function, which accepts two parameters, `target` and `observatory`:
                - `target`: an instance of `Target` (ie, this class). `self` will be passed.
                - `observatory`: `astroplan.Observer`, or `None`
            it should return one or three objects:
                - `score`: a float
                - `score_comments`: an iterable (list) of strings explaining the score.
                - `reject_comments`: an iterable (list) of strings explaing rejection, or `None`.

        """

        t_ref = t_ref or Time.time()
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None

        scoring_result = scoring_function(self, observatory)
        # Note that we're passing self as the first arg!

        error_msg = (
            f"your function {scoring_function.__name__} should return ",
            f"the score, and optionally two lists of strings",
            "`score_comments` and `reject_comments`",
        )
        if isinstance(scoring_result, float):
            score_value = scoring_result
            score_comments = None
            reject_comments = None
        elif isinstance(scoring_result, tuple):
            if len(scoring_result) != 3:
                raise ValueError(error_msg)
            score_value, score_comments, reject_comments = scoring_result
        else:
            raise ValueError(error_msg)

        self.score_comments[obs_name] = score_comments
        self.reject_comments[obs_name] = reject_comments

        if obs_name not in self.score_history:
            self.score_history[obs_name] = []

        score_tuple = (score_value, t_ref)
        self.score_history[obs_name].append(score_tuple)
        return score_value

    def check_night_relevance(self, t_ref: Time):
        pass
        # TODO: implement! check that the self.observatory_tonight() are relevant...

    def get_last_score(self, obs_name: str = None, return_time=False):
        """
        Provide a string (observatory name) and return.

        Parameters
        ----------
        obs_name: str
            the name of an observatory that the system knows about
        return_time: bool
            optional, defaults to `False`. If `False` return only the score
            (either a float or `None`).
            If `True`, return a tuple (score, astropy.time.Time), the time is
            the t_ref when the score was computed.
        """

        if obs_name is None:
            obs_name = "no_observatory"
        if isinstance(obs_name, Observer):
            obs_name = obs_name.name
            msg = (
                f"you should provide a string observatory name (eg. {obs_name}), "
                "not the `astroplan.Observer`"
            )
            warnings.warn(msg, Warning)

        obs_score_history = self.score_history.get(obs_name, [])

        if len(obs_score_history) == 0:
            result = (None, None)
        else:
            result = obs_score_history[-1]

        if return_time:
            return result
        else:
            return result[0]
