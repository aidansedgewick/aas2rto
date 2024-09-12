import copy
import time
import traceback
import warnings
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import Table, vstack
from astropy.table import unique as unique_table
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from astroplan import FixedTarget, Observer
from astroplan.plots import plot_altitude

from aas2rto.exc import MissingDateError, UnknownObservatoryWarning
from aas2rto.target_data import TargetData
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.utils import get_observatory_name

logger = getLogger(__name__.split(".")[-1])

matplotlib.use("Agg")

DEFAULT_ZTF_BROKER_PRIORITY = ("fink", "alerce", "lasair", "antares")


class Target:
    """
    TODO: docstring here!
    """

    def __init__(
        self,
        objectId: str,
        ra: float,
        dec: float,
        target_data: Dict[str, TargetData] = None,
        base_score: float = 1.0,
        target_of_opportunity: bool = False,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        # Basics
        self.objectId = objectId
        self.update_coordinates(ra, dec)
        self.base_score = base_score
        self.compiled_lightcurve = None

        # Target data
        self.target_data = target_data or {}

        # Observatory data
        self.observatory_info = {"no_observatory": None}

        # Models
        self.models = {}
        self.models_t_ref = {}

        # Scoring data
        self.score_history = {"no_observatory": []}
        self.score_comments = {"no_observatory": []}
        self.reject_comments = {"no_observatory": []}
        self.rank_history = {"no_observatory": []}

        # Keep track of what's going on
        self.creation_time = t_ref
        self.target_of_opportunity = target_of_opportunity
        self.alternative_objectIds = {}
        self.updated = False
        self.to_reject = False
        self.send_updates = False
        self.update_messages = []
        self.sudo_messages = []

    def __str__(self):
        if self.ra is None or self.dec is None:
            return f"{self.objectId}: NO COORDINATES FOUND!"
        return f"{self.objectId}: ra={self.ra:.5f} dec={self.dec:.5f}"

    def update_coordinates(self, ra: float, dec: float):
        self.ra = ra
        self.dec = dec
        self.coord = None
        self.astroplan_target = None
        if ra is not None and dec is not None:
            self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            self.astroplan_target = FixedTarget(self.coord, self.objectId)  # for plots?
        else:
            logger.warning(f"{self.objectId}: ra={ra} or dec={dec} is None!")

    def get_target_data(self, source):
        """
        Return the TargetData associated with a certain source.
        If the target_data does not exist, create a new one, and return that.

        Parameters
        ----------
        source [str]:
            the name of the data source to get.

        eg. with an existing TargetData
        >>> t1 = Target("target_01", ra=45., dec=60.)
        >>> cool_parameters = dict(a=10, b=20)
        >>> t1.target_data["cool_source"] = TargetData(parameters=cool_parameters)
        >>> "cool_source" in t1.target_data
        True
        >>> cool_data = t1.get_target_data("cool_source")
        >>> cool_data.parameters["a"] == 10
        True

        eg. without target data
        >>> t2 = Target("target_02", ta=60., dec=30.)
        >>> "other_source" in t.target_data
        False
        >>> other_data = t2.get_target_data("other_source")
        >>> isinstance(other_data, TargetData)
        True
        >>> "other_source" in t2.target_data
        True
        """

        source_data = self.target_data.get(source, None)
        if source_data is None:
            source_data = TargetData()
            self.target_data[source] = source_data
        return source_data

    def update_score_history(
        self,
        score_value: float,
        observatory: Observer = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        obs_name = get_observatory_name(observatory)
        if obs_name not in self.score_history:
            # Make sure we can append the score
            self.score_history[obs_name] = []

        score_tuple = (score_value, t_ref)
        self.score_history[obs_name].append(score_tuple)
        return

    def get_score_history(self, t_ref: Time = None):

        row_list = []
        for obs, obs_score_history in self.score_history.items():
            for data_tuple in obs_score_history:
                data = dict(observatory=obs, mjd=data_tuple[1].mjd, score=data_tuple[0])
                row_list.append(data)
        if len(row_list) == 0:
            return
        score_history_df = pd.DataFrame(row_list)
        score_history_df.sort_values(
            ["observatory", "mjd"], inplace=True, ignore_index=True
        )
        return score_history_df

    def update_rank_history(
        self,
        rank: int,
        observatory: Observer = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        obs_name = get_observatory_name(observatory)
        if obs_name not in self.rank_history:
            # Make sure we can append the rank
            self.rank_history[obs_name] = []

        rank_tuple = (rank, t_ref)
        self.rank_history[obs_name].append(rank_tuple)
        return

    def get_rank_history(self, t_ref: Time = None):
        row_list = []
        for obs, obs_rank_history in self.rank_history.items():
            for data_tuple in obs_rank_history:
                data = dict(
                    observatory=obs, mjd=data_tuple[1].mjd, ranking=data_tuple[0]
                )
                row_list.append(data)
        if len(row_list) == 0:
            return
        rank_history_df = pd.DataFrame(row_list)
        rank_history_df.sort_values(
            ["observatory", "mjd"], inplace=True, ignore_index=True
        )
        return rank_history_df

    def get_last_score(
        self, observatory: Union[Observer, str] = None, return_time=False
    ):
        """
        Provide a string (observatory name) and return.

        Parameters
        ----------
        observatory: `astroplan.Observer` | `None` | `str` [optional]
            default=None (="no_observatory")
            an observatory that the system knows about.
        return_time: `bool`
            optional, defaults to `False`. If `False` return only the score
            (either a float or `None`).
            If `True`, return a tuple (score, astropy.time.Time), the time is
            the t_ref when the score was computed.
        """

        obs_name = get_observatory_name(observatory)  # Returns "no_observatory" if None

        if obs_name not in self.score_history.keys():
            msg = f"No scores for observatory {obs_name}. Known: {self.score_history.keys()}"
            warnings.warn(UnknownObservatoryWarning(msg))

        obs_history = self.score_history.get(obs_name, [])
        if len(obs_history) == 0:
            result = (None, None)
        else:
            result = obs_history[-1]

        if return_time:
            return result
        return result[0]  # Otherwise just return the score.

    def get_info_string(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_ref_str = t_ref.strftime("%Y-%m-%d %H:%M")

        lines = [f"Target {self.objectId} at {t_ref_str}, see:"]
        broker_lines = (
            f"    FINK: fink-portal.org/{self.objectId}\n"
            f"    Lasair: lasair-ztf.lsst.ac.uk/objects/{self.objectId}\n"
            f"    ALeRCE: alerce.online/object/{self.objectId}"
        )
        lines.append(broker_lines)

        tns_data = self.target_data.get("tns", None)
        if tns_data:
            name = tns_data.parameters.get("Name", None)
            if name is not None:
                tns_name = tns_data.parameters["Name"]
                tns_code = tns_name.split()[1]
                lines.append(f"    TNS: wis-tns.org/object/{tns_code}")

        lines.append("coordinates:")
        if self.ra is not None and self.dec is not None:
            eq_line = f"    equatorial (ra, dec) = ({self.ra:.4f},{self.dec:+.5f})"
            lines.append(eq_line)
        if self.coord is not None:
            gal = self.coord.galactic
            gal_line = f"    galactic (l, b) = ({gal.l.deg:.4f},{gal.b.deg:+.5f})"
            lines.append(gal_line)
        if self.compiled_lightcurve is not None:
            ndet = {}
            last_mag = {}
            if "band" in self.compiled_lightcurve.columns:
                for band, band_history in self.compiled_lightcurve.groupby("band"):
                    if "tag" in band_history.columns:
                        detections = band_history[band_history["tag"] == "valid"]
                    else:
                        detections = band_history
                    if len(detections) > 0:
                        ndet[band] = len(detections)
                        last_mag[band] = detections["mag"].iloc[-1]
            if len(last_mag) > 0:
                magvals_str = ", ".join(f"{k}={v:.2f}" for k, v in last_mag.items())
                comm_line = "    last " + magvals_str
                lines.append(comm_line)
            if len(ndet) > 0:
                l = (
                    "    "
                    + ", ".join(f"{v} {k}" for k, v in ndet.items())
                    + " detections"
                )
                lines.append(l)

        return "\n".join(lines)

    def write_comments(self, outdir: Path, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.debug(f"{self.objectId}: writing comments")

        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        missing_score_comments = ["no score_comments provided"]
        missing_reject_comments = ["no reject_comments provided"]

        lines = self.get_info_string(t_ref=t_ref).split("\n")
        last_score = self.get_last_score()
        if last_score is None:
            score_str = "no_score"
        else:
            score_str = f"{last_score:.3f}"
        lines.append(f"score = {score_str} for no_observatory")

        score_comments = self.score_comments.get("no_observatory", None)
        score_comments = score_comments or missing_score_comments
        for comm in score_comments:
            lines.append(f"    {comm}")

        for obs_name, comments in self.score_comments.items():
            if obs_name == "no_observatory":  # We've just done this.
                continue
            if comments is None:
                continue
            obs_comments = [comm for comm in comments if comm not in score_comments]
            if len(obs_comments) == 0:
                continue
            obs_last_score = self.get_last_score(obs_name)
            if obs_last_score is None:
                continue
            lines.append(f"score = {obs_last_score:.3f} for {obs_name}")
            for comm in obs_comments:
                lines.append(f"    {comm}")

        if last_score is not None:
            if self.to_reject or not np.isfinite(last_score):
                lines.append(f"{self.objectId} rejected at {t_ref.iso}")
                reject_comments = self.reject_comments.get("no_observatory", None)
                reject_comments = reject_comments or missing_reject_comments
                for comm in reject_comments:
                    lines.append(f"    {comm}")

        comments_file = outdir / f"{self.objectId}_comments.txt"
        with open(comments_file, "w+") as f:
            f.writelines([l + "\n" for l in lines])
        return lines
