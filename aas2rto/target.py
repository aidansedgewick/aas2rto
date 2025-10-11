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
    The main class which holds information about a single (transient)
    target/object/candidate.

    Parameters
    ----------
    target_id: str
        the main name your object should be called by.
    ra: float
        Right Ascension, in decimal degrees.
    dec: float
        Declination, in decimal degrees.
    target_data: Dict[str, aas2rto.target_data.TargetData]
        A lookup of data from various sources.
        An entry for `fink` data, one for `atlas` data, one for `tns` data, etc...
    source:
        which data created this target? will be added into alt_ids. eg.
        >>> t = Target("T101", 30.0, 30.0, source="src01")
        >>> "src01" in t.alt_ids # True
    alt_ids: Dict [str, str]
        Lookup to keep track of other names for this target.
        eg. ZTF24abcdef might also be called SN 24abc.
        So alt_ids might be: `{'tns': 'SN 24abc'}`
        It's useful to keep track of these to keep updating data from other surveys.
    base_score: float, default=1.0
        The 'unmodified score'.
    target_of_opportunity: bool, default=False
        If target_of_opportunity, this target will never be rejected/removed from set of
        unranked targets.
        But - it might not appear in all ranked lists (due to eg. visibility criteria).
    t_ref: astropy.time.Time, default=Time.now()
        Recorded in target.creation_time.
    """

    default_base_score = 1.0

    def __init__(
        self,
        target_id: str,
        coord: SkyCoord,
        target_data: Dict[str, TargetData] = None,
        source: str = None,
        alt_ids: Dict[str, str] = None,
        base_score: float = None,
        target_of_opportunity: bool = False,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        # Basics
        self.target_id = target_id

        self.update_coordinates(coord)
        self.base_score = base_score or self.default_base_score
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
        self.rank_history = {"no_observatory": []}

        # Paths to scratch figures
        self.lc_fig_path = None
        self.vis_fig_paths = {}
        self.additional_fig_paths = {}

        # Is this target known by any other names?
        self.alt_ids = alt_ids or {}
        if source is not None:
            self.alt_ids[source] = target_id
        else:
            if target_id not in self.alt_ids.values():
                self.alt_ids["<unknown>"] = target_id

        # Keep track of what's going on
        self.creation_time = t_ref
        self.target_of_opportunity = target_of_opportunity
        self.updated = False
        self.to_reject = False
        self.send_updates = False
        self.update_messages = []
        self.sudo_messages = []

    # def update_coordinates(self, ra: float, dec: float):
    #     self.ra = ra
    #     self.dec = dec
    #     self.coord = None
    #     self.astroplan_target = None
    #     if ra is not None and dec is not None:
    #         self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    #         self.astroplan_target = FixedTarget(
    #             self.coord, self.target_id
    #         )  # for plots?
    #     else:
    #         logger.warning(f"{self.target_id}: ra={ra} or dec={dec} is None!")

    def update_coordinates(self, coord: SkyCoord):
        self.coord = coord
        # self.astroplan_target = FixedTarget(self.coord, self.target_id)

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

        score_tuple = (score_value, t_ref.mjd)
        self.score_history[obs_name].append(score_tuple)
        return

    def get_score_history(self, observatory=False, t_ref: Time = None):
        """
        Returns df with three columns: 'observatory', 'mjd', 'score'
            (avoid column name 'rank' as clash with pd keywords...)
        """

        if observatory is False:
            obs_names = list(self.score_history.keys())
        else:
            if not isinstance(observatory, list):
                observatory = [observatory]
            obs_names = [get_observatory_name(o) for o in observatory]

        df_list = []
        for obs_name in obs_names:
            obs_score_history = self.score_history.get(obs_name, None)
            if obs_score_history is None:
                msg = f"{self.target_id} has no score_history for {obs_name}"
                logger.warning(msg)
                warnings.warn(UnknownObservatoryWarning(msg))
                continue

            obs_df = pd.DataFrame(obs_score_history, columns=["score", "mjd"])
            obs_df["observatory"] = obs_name
            df_list.append(obs_df)

        if len(df_list) == 0:
            return pd.DataFrame(columns=["score", "mjd", "observatory"])
        score_history_df = pd.concat(df_list)
        score_history_df.sort_values(
            ["observatory", "mjd"], inplace=True, ignore_index=True
        )
        if t_ref is not None:
            score_history_df = score_history_df[score_history_df["mjd"] < t_ref.mjd]
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

        rank_tuple = (rank, t_ref.mjd)
        self.rank_history[obs_name].append(rank_tuple)
        return

    def get_rank_history(self, observatory=False, t_ref: Time = None):
        """
        Returns df with three columns: 'observatory', 'mjd', 'ranking'
            (avoid column name 'rank' as clash with pd keywords...)
        """

        if observatory is False:
            obs_names = list(self.score_history.keys())
        else:
            if not isinstance(observatory, list):
                observatory = [observatory]
            obs_names = [get_observatory_name(o) for o in observatory]

        df_list = []
        for obs_name in obs_names:
            obs_data = self.rank_history.get(obs_name, None)
            if obs_data is None:
                msg = f"{self.target_id} has no rank history for '{obs_name}'"
                logger.warning(msg)
                warnings.warn(UnknownObservatoryWarning(msg))
                continue
            obs_df = pd.DataFrame(obs_data, columns=["ranking", "mjd"])
            obs_df["observatory"] = obs_name
            df_list.append(obs_df)

        if len(df_list) == 0:
            return pd.DataFrame(columns=["ranking", "mjd", "observatory"])
        rank_history_df = pd.concat(df_list)
        rank_history_df.sort_values(
            ["observatory", "mjd"], inplace=True, ignore_index=True
        )
        if t_ref is not None:
            rank_history_df = rank_history_df[rank_history_df["mjd"] < t_ref.mjd]
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

    def get_last_rank(
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

        rank_history = self.rank_history.get(obs_name, [])
        if len(rank_history) == 0:
            result = (None, None)
        else:
            result = rank_history[-1]

        if return_time:
            return result
        return result[0]  # Otherwise just return the score.

    def get_info_lines(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_ref_str = t_ref.strftime("%Y-%m-%d %H:%M")

        info_lines = [f"Target {self.target_id} at {t_ref_str}"]

        target_id_lines = self.get_target_id_info_lines()
        coordinate_lines = self.get_coordinate_info_lines()
        photometry_lines = self.get_photometry_info_lines()

        info_lines.extend(target_id_lines)
        info_lines.extend(coordinate_lines)
        info_lines.extend(photometry_lines)
        return info_lines

    def get_target_id_info_lines(self):
        info_lines = []
        info_lines.append("Aliases and brokers")
        broker_name = self.alt_ids.get("ztf", None)
        if broker_name is not None:
            broker_lines = [
                f"    FINK: fink-portal.org/{broker_name}",
                f"    Lasair: lasair-ztf.lsst.ac.uk/objects/{broker_name}",
                f"    ALeRCE: alerce.online/object/{broker_name}",
            ]
            info_lines.extend(broker_lines)

        tns_name = self.alt_ids.get("tns", None)
        if tns_name is not None:
            tns_url = f"wis-tns.org/object/{tns_name}"
            info_lines.append(f"    TNS: {tns_url}")

        yse_name = self.alt_ids.get("yse", None)
        if yse_name is not None:
            yse_url = f"ziggy.ucolick.org/yse/transient_detail/{yse_name}"
            info_lines.append(f"    YSE: {yse_url}")

        alt_rev = {}
        for source, alt_name in self.alt_ids.items():
            if alt_name not in alt_rev:
                alt_rev[alt_name] = [source]
            else:
                alt_rev[alt_name].append(source)

        info_lines.append("alt names")
        for name, source_list in alt_rev.items():
            l = f"    {name} (" + ",".join(source_list) + ")"
            info_lines.append(l)

        return info_lines

    def get_coordinate_info_lines(self):
        info_lines = []
        info_lines.append("Coordinates:")
        if self.coord is not None:
            eq_str = f"{self.coord.ra.deg:.4f},{self.coord.dec.deg:+.5f}"
            eq_line = f"    equatorial (ra, dec) = {eq_str}"
            info_lines.append(eq_line)
        if self.coord is not None:
            gal = self.coord.galactic
            gal_line = f"    galactic (l, b) = ({gal.l.deg:.4f},{gal.b.deg:+.5f})"
            info_lines.append(gal_line)
        return info_lines

    def get_photometry_info_lines(self):
        info_lines = []
        info_lines.append("Photometry")
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
                mag_line = "    last " + magvals_str
                info_lines.append(mag_line)
            if len(ndet) > 0:
                l = (
                    "    "
                    + ", ".join(f"{v} {k}" for k, v in ndet.items())
                    + " detections"
                )
                info_lines.append(l)
        else:
            info_lines.append("    no photometry available")
        return info_lines

    def get_info_string(self, t_ref: Time = None):
        info_lines = self.get_info_lines(t_ref=t_ref)
        return "\n".join(info_lines)

    def write_comments(self, outdir: Path, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.debug(f"{self.target_id}: writing comments")

        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        missing_score_comments = ["no score_comments provided"]

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
                lines.append(f"{self.target_id} rejected at {t_ref.iso}")

        comments_file = outdir / f"{self.target_id}_comments.txt"
        with open(comments_file, "w+") as f:
            f.writelines([l + "\n" for l in lines])
        return lines
