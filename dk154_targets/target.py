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

# try:
#    import pygtc
# except ModuleNotFoundError as e:
#    pygtc = None
try:
    import corner
except ModuleNotFoundError as e:
    # print("\033[31;1mNo module corner\033[0m")
    corner = None

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
        include_badqual=True,
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
        self, lightcurve: pd.DataFrame, tag_col="tag", include_badqual=True
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

        if keep_updated:
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
        raise ValueError("should noth have made it here!")

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

    default_base_score = 100.0

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

        # Where are plots saved?
        self.latest_lc_fig_path = None
        self.lc_plot_t_ref = None
        self.latest_oc_fig_paths = {}
        self.oc_plot_t_ref = None

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

    def update_rank_history(self, rank: int, observatory: Observer, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        obs_name = get_observatory_name(observatory)
        if obs_name not in self.rank_history:
            # Make sure we can append the rank
            self.rank_history[obs_name] = []

        rank_tuple = (rank, t_ref)
        self.rank_history[obs_name].append(rank_tuple)
        return

    def check_night_relevance(self, t_ref: Time):
        pass
        # TODO: implement! check that the self.observatory_tonight() are relevant...

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

    def evaluate_target(
        self,
        scoring_function: Callable,
        observatory: Observer,
        t_ref: Time = None,
    ):
        """
        Evaluate this Target according to your scoring function.

        the score, and any scoring_comments or reject_comments are accessible
        >>> t = Target("T101", ra=10., dec=20.)
        >>> def scoring_function(target, obs, t_ref):
        ...     return 40., ["my comment"], []
        >>> obs = Observer(EarthLocation.of_site("lasilla"), name="lasilla")
        >>> t.evaluate_target(scoring_function, obs)
        >>> t.get_last_score(obs) == 40
        True
        >>> t.score_history["lasilla"]
        [(<first_score>, <time-then>), ..., (40., <time_now>)]
        >>> t.score_comments["lasilla"]
        ["my comment"]


        Parameters
        ----------
        scoring_function [Callable]
            a callable function, which accepts three parameters, `target`, `observatory` and `t_ref`:
                - `target`: an instance of `Target` (ie, this class).
                - `observatory`: `astroplan.Observer`, or `None`. Your function should be able to handle observatory=None.
                - `t_ref`: an `astropy.time.Time`
            it should return one or three objects:
                - `score`: a float
                - `score_comments`: an iterable (list) of strings explaining the score.
                - `reject_comments`: an iterable (list) of strings explaing rejection, or `None`.
        observatory [`astroplan.Observer` | None]
            the observatory to observer from. It can be none.
        t_ref [`astropy.time.Time`] (optional, default=Now)
            The time that the score is evaluated at.

        Returns
        -------
        score
            the value produced by the scoring_function
        """

        t_ref = t_ref or Time.now()

        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None

        try:
            func_name = scoring_function.__name__
        except AttributeError:
            try:
                func_name = type(scoring_function).__name__
            except Exception:
                func_name = "func_name"

        if isinstance(scoring_function, type):
            msg = (
                f"\n    It looks like your scoring function is a class, '{func_name}'"
                f"\n    You should initialise it first - ie. `\033[33;1mfunc = {func_name}()\033[0m`"
            )  # In 2nd string, () are DEFINITELY outside of the {}. We want literal () brackets.
            warnings.warn(UserWarning(msg))
            try:
                scoring_function = scoring_function()
            except Exception as e:
                pass

        scoring_result = scoring_function(self, observatory, t_ref)
        # `self` is first arg -- ie THIS target!!

        ##===== TODO: Check signature of function! =====##
        error_msg = (
            f"Your function '{func_name}' has returned:\n    {scoring_result}\n    type {type(scoring_result)}\n"
            f"It should return the `score` (a single `float` or `int`),\n"
            f"OR the score and two lists of strings: `score_comments` and `reject_comments`"
        )
        if isinstance(scoring_result, float) or isinstance(scoring_result, int):
            score_value = scoring_result
            score_comments = []
            reject_comments = []
        elif isinstance(scoring_result, tuple):
            if len(scoring_result) != 3:
                raise ValueError(error_msg)
            score_value, score_comments, reject_comments = scoring_result
        else:
            raise ValueError(error_msg)

        self.score_comments[obs_name] = score_comments
        self.reject_comments[obs_name] = reject_comments
        self.update_score_history(score_value, observatory, t_ref=t_ref)

        return score_value

    def build_compiled_lightcurve(
        self,
        lightcurve_compiler: Callable,
        t_ref: Time = None,
        lazy=False,
    ):
        """
        Compile all the data from the target_data into a convenient location,
        This could be useful for eg. scoring, modeling, plotting.

        Parameters
        ----------
        lightcurve_compiler [Callable]
            your function that builds a convenient single lightcurve.
            see dk154_targets.lightcurve_compilers.DefaultLigthcurveCompiler for example.
        lazy [bool] default=True

        Note: the name of this method is a bit silly, but the better option 'compile_lightcurve'
        is only one character away from "compiled_lightcurve", which is the actual attr...
        """
        t_ref = t_ref or Time.now()

        if lazy and not self.updated and self.compiled_lightcurve is not None:
            logger.debug(f"{self.objectId}: skip compile, not updated and lazy=True")
            return

        if lightcurve_compiler is None:
            logger.warning("no light curve compiled.")
            raise ValueError("lightcurve compiler cannot be None.")

        compiled_lightcurve = lightcurve_compiler(self)
        if compiled_lightcurve is None:
            logger.warning(f"{self.objectId} compiled lightcurve is None")
        self.compiled_lightcurve = compiled_lightcurve
        return compiled_lightcurve

    def build_model(self, modeling_function: Callable, t_ref: Time = None):
        """
        Build a target model and store it, so you can access it later for storing/
        plotting, etc.

        >>> from my_modeling_functions import amazing_sn_model
        >>> my_model = target.build_model(amazing_sn_model)

        you can also then access with
        >>> my_model = target.models["amazing_sn_model"]
        >>> t_built = target.models_t_ref["amazing_sn_model"]
        >>> print(f"{type(my_model)} model built at {t_built.isot}")
        SuperAmazingSupernovae model built at 2023-02-25T00:00:00.000

        Parameters
        ----------
        modeling_function
            your function, which should accept one argument, `Target` and return
            a model (which you can access later)
        t_ref
            an `astropy.time.Time`

        """
        t_ref = t_ref or Time.now()

        try:
            model_key = modeling_function.__name__
        except AttributeError as e:
            class_name = type(modeling_function).__name__
            msg = f"\n    Your modeling_function {class_name} should have attribute __name__."
            raise ValueError(msg)

        try:
            model = modeling_function(self)
        except Exception as e:
            logger.warning(f"error modeling {self.objectId} with {model_key}:")
            tr = traceback.format_exc()
            print(tr)
            model = None

        self.models[model_key] = model
        self.models_t_ref[model_key] = t_ref
        return model

    def plot_lightcurve(
        self,
        t_ref: Time = None,
        plotting_function: Callable = None,
        fig_path=None,
    ) -> plt.Figure:
        t_ref = t_ref or Time.now()

        if plotting_function is None:
            plotting_function = plot_default_lightcurve
        try:
            print("TRY PLOTTING FIG")
            lc_fig = plotting_function(self, t_ref=t_ref)
        except Exception as e:
            print(e)
            lc_fig = None
        if fig_path is not None and lc_fig is not None:
            print("TRY SAVING FIG")
            try:
                lc_fig.savefig(fig_path)
            except AttributeError:
                return lc_fig
            self.latest_lc_fig_path = fig_path
        else:
            print("FIG_PATH", fig_path)
        self.lc_fig_t_ref = t_ref
        return lc_fig

    def plot_observing_chart(
        self, observatory, t_ref: Time = None, fig_path=None, **kwargs
    ):
        obs_name = getattr(observatory, "name", "no_observatory")
        t_ref = t_ref or Time.now()

        try:
            get_observatory_name(observatory)
            obs_info = self.observatory_info.get(obs_name, None)
            oc_fig = plot_observing_chart(
                observatory, self, t_ref=t_ref, obs_info=obs_info, **kwargs
            )
        except Exception as e:
            print(traceback.format_exc())
            oc_fig = None
        if oc_fig is not None and fig_path is not None:
            oc_fig.savefig(fig_path)
            self.latest_oc_fig_paths[obs_name] = fig_path
        self.oc_plot_t_ref = t_ref
        return oc_fig

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
                        detections = band_history.query("tag=='valid'")
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

    def reset_figures(self):
        # self.latest_lc_fig = None
        self.latest_lc_fig_path = None
        # self.latest_oc_figs = {}
        self.latest_oc_fig_paths = {}


def plot_default_lightcurve(target: Target, return_plotter=False, t_ref: Time = None):
    t_ref = t_ref or Time.now()
    plotter = DefaultLightcurvePlotter.plot(target, t_ref=t_ref)
    if return_plotter:
        return plotter
    return plotter.fig


class DefaultLightcurvePlotter:
    lc_gs = plt.GridSpec(3, 4)
    zscaler = ZScaleInterval()

    figsize = (6.5, 5)

    ztf_colors = {"ztfg": "C0", "ztfr": "C1"}
    atlas_colors = {"atlasc": "C2", "atlaso": "C3"}
    plot_colors = {**ztf_colors, **atlas_colors, "no_band": "k"}

    det_kwargs = dict(ls="none", marker="o")
    ulim_kwargs = dict(ls="none", marker="v", mfc="none")
    badqual_kwargs = dict(ls="none", marker="o", mfc="none")

    tag_col = "tag"
    valid_tag = "valid"
    ulimit_tag = "upperlim"
    badqual_tag = "badquality"

    band_col = "band"

    @classmethod
    def plot(cls, target: Target, t_ref: Time = None) -> plt.Figure:
        t_ref = t_ref or Time.now()

        plotter = cls(t_ref=t_ref)
        plotter.init_fig()
        plotter.plot_photometry(target)
        plotter.add_cutouts(target)
        plotter.format_axes(target)
        plotter.add_comments(target)
        return plotter

    def __init__(self, t_ref: Time = None):
        self.t_ref = t_ref or Time.now()

        self.legend_handles = []
        self.peakmag_vals = []
        self.photometry_plotted = False
        self.cutouts_added = False
        self.axes_formatted = False
        self.comments_added = False

    def init_fig(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(self.lc_gs[:, :-1])

    def plot_photometry(self, target: Target, band_col=None):

        if target.compiled_lightcurve is None:
            logger.warning(
                f"{target.objectId} has no compiled lightcurve for plotting."
            )
            return self.fig
        lightcurve = target.compiled_lightcurve.copy()

        if "jd" not in lightcurve.columns:
            if "mjd" in lightcurve.columns:
                time_dat = Time(lightcurve["mjd"].values, format="mjd")
            else:
                msg = f"missing date column to plot lightcurve: {lightcurve.columns}"
                logger.error(msg)
                raise ValueError(msg)
            lightcurve.loc[:, "jd"] = time_dat.jd

        band_col = band_col or self.band_col
        if band_col not in lightcurve.columns:
            logger.error(f"no column {band_col} in compiled_lightcurve.columns")
            lightcurve.loc[:, band_col] = "no_band"

        for ii, (band, band_history) in enumerate(lightcurve.groupby(band_col)):

            band_color = self.plot_colors.get(band, f"C{ii%8}")
            band_kwargs = dict(color=band_color)

            scatter_handle = self.ax.errorbar(
                0, 0, yerr=0.1, label=band, **band_kwargs, **self.det_kwargs
            )
            self.legend_handles.append(scatter_handle)

            if self.tag_col in band_history.columns:
                detections = band_history[band_history[self.tag_col] == self.valid_tag]
                ulimits = band_history[band_history[self.tag_col] == self.ulimit_tag]
                badqual = band_history[band_history[self.tag_col] == self.badqual_tag]

                if len(ulimits) > 0:
                    xdat = ulimits["jd"].values - self.t_ref.jd
                    ydat = ulimits["diffmaglim"]
                    self.ax.errorbar(xdat, ydat, **band_kwargs, **self.ulim_kwargs)
                if len(badqual) > 0:
                    xdat = badqual["jd"].values - self.t_ref.jd
                    ydat = badqual["mag"].values
                    yerr = badqual["magerr"].values
                    self.ax.errorbar(
                        xdat, ydat, yerr=yerr, **band_kwargs, **self.badqual_kwargs
                    )
            else:
                detections = band_history

            if len(detections) > 0:
                xdat = detections["jd"] - self.t_ref.jd
                ydat = detections["mag"]
                yerr = detections["magerr"]
                self.ax.errorbar(
                    xdat, ydat, yerr=yerr, **band_kwargs, **self.det_kwargs
                )
                self.photometry_plotted = True

    def add_cutouts(self, target: Target):
        cutouts = {}
        for broker in DEFAULT_ZTF_BROKER_PRIORITY:
            source_data = target.target_data.get(broker, None)
            if source_data is None:
                continue
            if len(source_data.cutouts) == 0:
                continue
            cutouts = source_data.cutouts
            break

        for ii, imtype in enumerate(["Science", "Template", "Difference"]):
            im_ax = self.fig.add_subplot(self.lc_gs[ii : ii + 1, -1:])

            im_ax.set_xticks([])
            im_ax.set_yticks([])

            imtext_kwargs = dict(
                rotation=90,
                transform=im_ax.transAxes,
                ha="left",
                va="center",
                fontsize=12,
            )
            im_ax.text(1.05, 0.5, imtype, **imtext_kwargs)

            im = cutouts.get(imtype.lower(), None)
            if im is None:
                continue

            im_finite = im[np.isfinite(im)]
            vmin, vmax = self.zscaler.get_limits(im_finite.flatten())
            im_ax.imshow(im, vmin=vmin, vmax=vmax)

            xl_im = len(im.T)
            yl_im = len(im)
            # add crosshairs
            im_ax.plot(
                [0.5 * xl_im, 0.5 * xl_im], [0.2 * yl_im, 0.4 * yl_im], color="r"
            )
            im_ax.plot(
                [0.2 * yl_im, 0.4 * yl_im], [0.5 * yl_im, 0.5 * yl_im], color="r"
            )

            self.cutouts_added = True

    def format_axes(self, target):
        title = str(target)
        tns_data = target.target_data.get("tns", None)
        if tns_data is not None:
            known_redshift = tns_data.parameters.get("Redshift", None)
            if known_redshift is not None:
                title = title + r" ($z_{\rm TNS}=" + f"{known_redshift:.3f}" + "$)"
        self.ax.text(
            0.5,
            0.98,
            title,
            fontsize=14,
            ha="center",
            va="top",
            transform=self.fig.transFigure,
        )

        self.peakmag_vals.append(17.2)
        y_bright = np.nanmin(self.peakmag_vals) - 0.2
        y_faint = 22.0
        self.ax.set_ylim(y_faint, y_bright)
        self.ax.axvline(0, color="k")

        legend = self.ax.legend(handles=self.legend_handles, loc=2)
        self.ax.add_artist(legend)
        date_str = self.t_ref.strftime("%d-%b-%y %H:%M")
        xlabel = f"Days before {date_str}"
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel("Difference magnitude", fontsize=12)
        try:
            self.add_readable_dateticks()
        except Exception as e:
            pass
        self.axes_formatted = True

    def add_readable_dateticks(self):
        twiny = self.ax.twiny()
        twiny.set_xlim(self.ax.get_xlim())

        x0, x1 = self.ax.get_xlim()
        s = 10
        xmin = np.sign(x0) * np.floor(abs(x0) / s) * s
        xmax = np.sign(x1) * np.ceil(abs(x1) / s) * s
        xticks = self.t_ref.jd + np.arange(xmin, xmax, s)
        xticklabels = [Time(int(x), format="jd").strftime("%d %b") for x in xticks]
        twiny.set_xticks(xticks - self.t_ref.jd)
        twiny.set_xticklabels(xticklabels)

    def add_comments(self, target):
        comments = target.score_comments.get("no_observatory", [])
        self.fig.subplots_adjust(bottom=0.3)
        if len(comments) > 0:
            N = len(comments) // 2
            text = "score comments:\n" + "\n".join(
                f"    {comm}" for comm in comments[:N]
            )
            self.fig.text(
                0.03, 0.2, text, ha="left", va="top", transform=self.fig.transFigure
            )
            text = "\n" + "\n".join(f"    {comm}" for comm in comments[N:])
            self.fig.text(
                0.53, 0.2, text, ha="left", va="top", transform=self.fig.transFigure
            )
            self.comments_added = True


def plot_observing_chart(
    observatory: Observer, *target_list: Target, return_plotter=False, **kwargs
):
    plotter = ObservingChartPlotter.plot(observatory, *target_list, **kwargs)
    if return_plotter:
        return plotter
    return plotter.fig


class ObservingChartPlotter:
    double_height_figsize = (6, 8)
    single_height_figsize = (6, 4)
    forecast = 1.0  # Days
    dt = 1 / (24 * 4)  # Days, equals 15 min

    moon_kwargs = {"color": "0.5", "ls": "--", "label": "moon"}
    sun_kwargs = {"color": "0.5", "label": "sun"}

    title_kwargs = {"fontsize": 14, "ha": "center", "va": "top"}

    # az_ticks = np.arange(0, 2 * np.pi, np.pi / 4)
    az_ticklabels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    @classmethod
    def plot(
        cls,
        observatory: Observer,
        *target_list: Target,
        alt_ax=True,
        sky_ax=True,
        sun=True,
        moon=True,
        obs_info: ObservatoryInfo = None,
        recompute=False,
        t_ref: Time = None,
        t_grid: Time = None,
    ):
        plotter = cls(observatory, obs_info=obs_info, t_ref=t_ref, t_grid=t_grid)
        plotter.init_fig(alt_ax=alt_ax, sky_ax=sky_ax)
        plotter.init_axes(alt_ax=alt_ax, sky_ax=sky_ax)

        for ii, target in enumerate(target_list):
            plotter.plot_target(target, recompute=recompute, idx=ii)
        if sun:
            plotter.plot_sun()
        if moon:
            plotter.plot_moon()
        plotter.format_axes()
        return plotter

    def __init__(
        self,
        observatory: Observer,
        obs_info: ObservatoryInfo = None,
        t_ref: Time = None,
        t_grid: Time = None,
    ):
        t_ref = t_ref or Time.now()

        self.observatory = observatory

        if obs_info is None:
            obs_info = ObservatoryInfo.for_observatory(
                self.observatory,
                t_grid=t_grid,
                t_ref=t_ref,
                dt=self.dt,
                forecast=self.forecast,
            )
        self.obs_info = obs_info
        self.t_grid = self.obs_info.t_grid

        self.legend_handles = []
        self.fig = None

        self.axes_initialized = False
        self.altitude_plotted = False
        self.sky_plotted = False
        self.moon_plotted = False
        self.sun_plotted = False
        self.axes_formatted = False

    def init_fig(self, alt_ax=True, sky_ax=True, figsize=None):
        if alt_ax and sky_ax:
            figsize = figsize or self.double_height_figsize
        else:
            figsize = figsize or self.single_height_figsize
        self.fig = plt.figure(figsize=figsize)
        self.alt_ax = None
        self.sky_ax = None

    def init_axes(self, alt_ax=True, sky_ax=True):
        if alt_ax and sky_ax:
            self.alt_ax = self.fig.add_axes(211)
            self.sky_ax = self.fig.add_axes(212, projection="polar")
        if alt_ax and (not sky_ax):
            self.alt_ax = self.fig.add_axes(111)
        if (not alt_ax) and sky_ax:
            self.sky_ax = self.fig.add_axes(111, projection="polar")
        self.axes_initialized = True

    def plot_target(self, target: Target, idx: int = None, recompute=False, **kwargs):
        obs_name = get_observatory_name(self.observatory)
        if isinstance(target, Target):
            coord = target.coord
            target_label = target.objectId
            obs_info = target.observatory_info.get(obs_name, None)
        elif isinstance(target, SkyCoord):
            coord = target
            target_label = None
            if idx is not None:
                target_label = f"target_{idx+1:02d}"
            obs_info = None
        else:
            raise TypeError(
                f"target should be `Target` or `SkyCoord`, not {type(target)}"
            )
        if obs_info is None or recompute:
            altaz = self.observatory.altaz(self.t_grid, coord)
        else:
            altaz = obs_info.target_altaz

        plot_kwargs = dict(
            label=target_label, color=f"C{idx%8}", ls=("-" if idx < 8 else "--")
        )
        plot_kwargs.update(kwargs)
        if self.alt_ax is not None:
            self.plot_altitude(self.t_grid, altaz, **plot_kwargs)
            self.altitude_plotted = True
        if self.sky_ax is not None:
            self.plot_sky(self.t_grid, altaz, **plot_kwargs)
            self.sky_plotted = True
        return

    def plot_altitude(self, t_grid: Time, altaz: AltAz, **kwargs):
        if self.alt_ax is None:
            msg = "alt_ax is None. either `plotter.init_axes(alt_ax=True)` or set plotter.alt_ax"
            raise ValueError(msg)
        line = self.alt_ax.plot(t_grid.mjd, altaz.alt.deg, **kwargs)
        self.legend_handles.append(line[0])
        return

    def plot_sky(self, t_grid, altaz: AltAz, **kwargs):
        if self.sky_ax is None:
            msg = "alt_ax is None. either `plotter.init_axes(alt_ax=True)` or set plotter.sky_ax"
            raise ValueError(msg)
        line = self.sky_ax.plot(altaz.az.rad, altaz.alt.deg, **kwargs)
        if self.alt_ax is None:
            self.legend_handles.append(line[0])
        return

    def plot_moon(self, **kwargs):
        moon_altaz = self.obs_info.moon_altaz
        if moon_altaz is None:
            moon_altaz = self.observatory.moon_altaz(self.t_grid)
        moon_kwargs = self.moon_kwargs.copy()
        moon_kwargs.update(kwargs)
        if self.alt_ax is not None:
            self.plot_altitude(self.t_grid, moon_altaz, **moon_kwargs)
        if self.sky_ax is not None:
            self.plot_sky(self.t_grid, moon_altaz, **moon_kwargs)
        self.moon_plotted = True
        return

    def plot_sun(self, **kwargs):
        sun_altaz = self.obs_info.sun_altaz
        if sun_altaz is None:
            sun_altaz = self.observatory.sun_altaz(self.t_grid)
        sun_kwargs = self.sun_kwargs.copy()
        sun_kwargs.update(kwargs)
        if self.alt_ax is not None:
            self.plot_altitude(self.t_grid, sun_altaz, **sun_kwargs)
        if self.sky_ax is not None:
            self.plot_sky(self.t_grid, sun_altaz, **sun_kwargs)
        self.sun_plotted = True
        return

    def format_axes(self):
        obs_name = self.observatory.name
        if obs_name is None:
            loc = self.observatory.location
            obs_name = f"{str(loc.lat.dms)}, {str(loc.lon.dms)}"
        t_str = self.t_grid[0].strftime("%Y-%m-%d %H:%M") + " UT"
        title = f"Observing from {obs_name} ({t_str})"
        self.fig.text(
            0.5, 0.98, title, transform=self.fig.transFigure, **self.title_kwargs
        )
        if self.alt_ax is not None:
            self.alt_ax.set_xlim(self.t_grid[0].mjd, self.t_grid[-1].mjd)
            self.alt_ax.set_ylim(-20.0, 90.0)
            self.alt_ax.axhline(0.0, color="k")
            self.alt_ax.set_ylabel(f"Altitude [deg]")
            try:
                self.set_readable_alt_xticks()
            except Exception as e:
                pass
            legend = self.alt_ax.legend(handles=self.legend_handles, ncols=3)
            self.alt_ax.add_artist(legend)
        if self.sky_ax is not None:
            self.sky_ax.set_rlim(bottom=90.0, top=0.0)
            self.sky_ax.set_rgrids((15.0, 30.0, 45.0, 60.0, 75.0))
            self.sky_ax.set_theta_zero_location("N")
            az_ticks = self.sky_ax.get_xticks()
            self.sky_ax.set_xticks(az_ticks)
            self.sky_ax.set_xticklabels(self.az_ticklabels)
        self.axes_formatted = True

    def set_readable_alt_xticks(self):
        n = 8
        delta = int(24 * (self.t_grid[-1].mjd - self.t_grid[0].mjd) / n)
        if delta < 1:
            return
        d = self.t_grid[0].ymdhms
        t0 = Time(dict(year=d.year, month=d.month, day=d.day, hour=int(d.hour)))
        vals = t0 + np.arange(1, n * delta, delta) * u.hour  # start=1 as 0 in past..
        ticks = vals[vals.mjd < self.t_grid[-1].mjd]
        tk0 = ticks[0].strftime("%H:%M\n%b %d")
        labels = [tk0]
        for tk in ticks[1:]:
            if tk.ymdhms.hour < delta:
                labels.append(tk.strftime("%H:%M\n%b %d"))
            else:
                labels.append(tk.strftime("%H:%M"))
        self.alt_ax.set_xticks(ticks.mjd)
        self.alt_ax.set_xticklabels(labels)
