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

from dk154_targets.exc import MissingDateError, UnknownObservatoryWarning
from dk154_targets.obs_info import ObservatoryInfo
from dk154_targets.utils import get_observatory_name

logger = getLogger(__name__.split(".")[-1])

matplotlib.use("Agg")

DEFAULT_ZTF_BROKER_PRIORITY = ("fink", "alerce", "lasair", "antares")


class SettingLightcurveDirectlyWarning(UserWarning):
    pass


class UnknownPhotometryTagWarning(UserWarning):
    pass


class TargetData:
    default_valid_tags = (
        "valid",
        "detection",
    )
    default_badqual_tags = (
        "badquality",
        "badqual",
        "dubious",
    )
    default_nondet_tags = (
        "upperlim",
        "nondet",
    )

    date_columns = ("jd", "mjd", "JD", "MJD")

    def __init__(
        self,
        lightcurve: pd.DataFrame = None,
        include_badqual=False,
        probabilities: pd.DataFrame = None,
        parameters: dict = None,
        cutouts: dict = None,
        meta: dict = None,
        valid_tags: tuple = default_valid_tags,
        badqual_tags: tuple = default_badqual_tags,
        nondet_tags: tuple = default_nondet_tags,
    ):
        self.meta = dict()
        meta = meta or {}

        self.valid_tags = valid_tags
        self.badqual_tags = badqual_tags
        self.nondet_tags = nondet_tags

        if lightcurve is not None:
            self.add_lightcurve(lightcurve.copy(), include_badqual=include_badqual)
        else:
            self.remove_lightcurve()
        self.probabilities = probabilities or {}
        self.parameters = parameters or {}

        self.cutouts = self.empty_cutouts()
        cutouts = cutouts or {}
        self.cutouts.update(cutouts)

    def __setattr__(self, name, value):
        if name == "lightcurve":
            msg = (
                "\nYou should use the targetdata.add_lightcurve(lc) method."
                "\nThis will correctly set the attributes "
                "`detections`, `badqual` and `non_detections` attributes,"
                "\nif the column `tag` is avalable."
            )
            warnings.warn(SettingLightcurveDirectlyWarning(msg))
        super().__setattr__(name, value)

    def add_lightcurve(
        self, lightcurve: pd.DataFrame, tag_col="tag", include_badqual=False
    ):
        lightcurve = lightcurve.copy()

        date_col = self.get_date_column(lightcurve)
        if isinstance(lightcurve, pd.DataFrame):
            lightcurve.sort_values(date_col, inplace=True)
        if isinstance(lightcurve, Table):
            lightcurve.sort(date_col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingLightcurveDirectlyWarning)
            self.lightcurve = lightcurve

        if include_badqual:
            detection_tags = self.valid_tags + self.badqual_tags
            badqual_tags = ()
        else:
            detection_tags = self.valid_tags
            badqual_tags = self.badqual_tags
        nondet_tags = self.nondet_tags
        all_tags = detection_tags + badqual_tags + nondet_tags
        if tag_col in self.lightcurve.columns:
            self.detections = self.lightcurve[
                np.isin(self.lightcurve[tag_col], detection_tags)
            ]
            self.badqual = self.lightcurve[
                np.isin(self.lightcurve[tag_col], badqual_tags)
            ]
            self.non_detections = self.lightcurve[
                np.isin(self.lightcurve[tag_col], nondet_tags)
            ]

            known_tag_mask = np.isin(lightcurve[tag_col], all_tags)
            if not all(known_tag_mask):
                unknown_tags = self.lightcurve[tag_col][~known_tag_mask]
                msg = f"\nin {tag_col}: {unknown_tags}\nexpected {all_tags}"
                warnings.warn(UnknownPhotometryTagWarning(msg))
        else:
            self.detections = self.lightcurve.copy()
            self.badqual = None
            self.non_detections = None
        return

    def remove_lightcurve(self):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=SettingLightcurveDirectlyWarning
            )  # TODO on 3.1
            self.lightcurve = None
        self.detections = None
        self.badqual = None
        self.non_detections = None

    def get_date_column(self, lightcurve):
        date_col = None
        for col in self.date_columns:
            if col in lightcurve.columns:
                date_col = col
                return date_col
        msg = (
            f"lightcurve columns {lightcurve.columns} should contain "
            f"a date column: {self.date_columns}"
        )
        raise MissingDateError(msg)

    def empty_cutouts(self) -> Dict[str, np.ndarray]:
        return {}

    def integrate_lightcurve_updates(
        self,
        updates: pd.DataFrame,
        column: str = "candid",
        continuous=False,
        keep_updates=True,
        **kwargs,
    ):
        """
        combine updates into a lightcurve.

        Parameters
        ----------
        updates
            pd.DataFrame, the updates you want to include in your updated lightcurve.
        column
            the column to check for matches, and check that repeated rows don't happen
        continuous [bool]
            default=False
            remove the end of the existing lightcurve, or the beginning of the updates
            (depending on keep_updates), and then concat
            otherwise remove duplicate values in column.
        **kwargs
            keyword arguments are passed to add_lightcurve
        """
        if updates is None:
            logger.warning("updates is None")
            return None

        if self.lightcurve is None:
            self.add_lightcurve(updates, **kwargs)
        if column not in self.lightcurve.columns:
            raise ValueError(f"{column} not in both lightcurve columns")
        if column not in updates.columns:
            raise ValueError(f"{column} not in both updates columns")
        if continuous:
            updated_lightcurve = self.integrate_continuous(
                updates, column, keep_updates=keep_updates
            )
        else:
            updated_lightcurve = self.integrate_equality(
                updates, column, keep_updates=keep_updates
            )
        self.add_lightcurve(updated_lightcurve, **kwargs)

    def integrate_continuous(self, updates, column, keep_updates=True):
        if not (isinstance(updates, pd.DataFrame) or isinstance(updates, Table)):
            raise TypeError(
                f"updates should be `pd.DataFrame` or `astropy.table.Table, "
                f"not {type(updates)}"
            )

        if keep_updates:
            existing_mask = self.lightcurve[column] < updates[column].min()
            existing = self.lightcurve[
                existing_mask
            ]  # Keep everything before updates start.
            # updates = updates # Obvious!
        else:
            existing = self.lightcurve
            updates_mask = updates[column] > self.lightcurve[column].max()
            updates = updates[updates_mask]

        updated_lightcurve = None
        if isinstance(updates, pd.DataFrame):
            return pd.concat([self.lightcurve, updates], ignore_index=True)
        if isinstance(updates, Table):
            return vstack([self.lightcurve, updates])
        raise ValueError("should not have made it here!")

    def integrate_equality(
        self, updates: pd.DataFrame, column, keep_updates=True, nan_values=(0,)
    ):
        """
        Concatenate existing lightcurve and updates, remove duplicated values.
        Which duplicated values are removed depends on `keep_updates`.

        Parameters
        ----------
        updates [`pd.DataFrame` or `astropy.table.Table`]
            the updates to include.
        """

        keep = "last" if keep_updates else "first"
        updated_lightcurve = None
        if isinstance(updates, pd.DataFrame):
            if updates is None:
                updated_lightcurve = self.lightcurve
            else:
                updated_lightcurve = pd.concat(
                    [self.lightcurve, updates], ignore_index=True
                )
            updated_lightcurve.drop_duplicates(
                subset=column, keep=keep, inplace=True, ignore_index=True
            )
        if isinstance(updates, Table):
            concat_lightcurve = vstack([self.lightcurve, updates])
            updated_lightcurve = unique_table(concat_lightcurve, keys=column, keep=keep)
        if updated_lightcurve is None:
            raise TypeError(
                f"updates should be `pd.DataFrame` or `astropy.table.Table, not {type(updates)}"
            )
        return updated_lightcurve


class Target:
    """
    TODO: docstring here!
    """

    default_base_score = 1.0

    def __init__(
        self,
        objectId: str,
        ra: float,
        dec: float,
        target_data: Dict[str, TargetData] = None,
        base_score: float = None,
        target_of_opportunity: bool = False,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        # Basics
        self.objectId = objectId
        self.update_coordinates(ra, dec)
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
        self.reject_comments = {"no_observatory": []}
        self.rank_history = {"no_observatory": []}

        # Keep track of what's going on
        self.creation_time = t_ref
        self.target_of_opportunity = target_of_opportunity
        self.updated = False
        self.to_reject = False
        self.send_updates = False
        self.update_messages = []
        self.sudo_messages = []

    def __str__(self):
        if self.ra is None or self.dec is None:
            return f"{self.objectId}: NO COORDINATES FOUND!"
        return f"{self.objectId}: ra={self.ra:.5f} dec={self.dec:.5f}"

    def update_coordinates(self, ra, dec):
        self.ra = ra
        self.dec = dec
        self.coord = None
        self.astroplan_target = None
        if ra is not None and dec is not None:
            self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            self.astroplan_target = FixedTarget(self.coord, self.objectId)  # for plots

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
        self, score_value: float, observatory: Observer, t_ref: Time = None
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

    def update_rank_history(self, rank: int, observatory: Observer, t_ref: Time = None):
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
            msg = f"Unknown observatory name {obs_name}. Known: {self.score_history.keys()}"
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
