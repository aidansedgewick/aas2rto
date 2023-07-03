import traceback
import warnings
from logging import getLogger
from typing import Callable, List

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from astroplan import FixedTarget, Observer
from astroplan.plots import plot_altitude

logger = getLogger(__name__.split(".")[-1])


class TargetData:
    valid_tags = ("valid",)
    badqual_tags = ("badquality", "badqual", "dubious")
    nondet_tags = ("upperlim", "nondet")

    def __init__(
        self,
        lightcurve: pd.DataFrame = None,
        probabilities: pd.DataFrame = None,
        parameters: dict = None,
        cutouts: dict = None,
        meta: dict = None,
    ):
        self.meta = dict()
        meta = meta or {}
        self.meta.update(meta)
        # self.updated = False

        if lightcurve is not None:
            self.add_lightcurve(lightcurve.copy())
        else:
            self.lightcurve = None
            self.detections = None
            self.non_detections = None
        self.probabilities = probabilities
        self.parameters = parameters or {}
        self.cutouts = cutouts or {}

    def add_lightcurve(
        self, lightcurve: pd.DataFrame, tag_col="tag", include_badqual=True
    ):
        if ("jd" not in lightcurve.columns) and ("mjd" not in lightcurve.columns):
            raise ValueError("lightcurve should have at least 'jd' or 'mjd'")
        if ("jd" in lightcurve.columns) and ("mjd" not in lightcurve.columns):
            mjd_dat = Time(lightcurve["jd"].values, format="jd").mjd
            jd_loc = lightcurve.columns.get_loc("jd")
            lightcurve.insert(jd_loc + 1, "mjd", mjd_dat)
        if ("mjd" in lightcurve.columns) and ("jd" not in lightcurve.columns):
            jd_dat = Time(lightcurve["mjd"].values, format="mjd").jd
            mjd_loc = lightcurve.columns.get_loc("mjd")
            lightcurve.insert(mjd_loc, "jd", jd_dat)

        lightcurve.sort_values("jd", inplace=True)
        self.lightcurve = lightcurve

        if include_badqual:
            det_query = " or ".join(
                f"tag=='{x}'" for x in self.valid_tags + self.badqual_tags
            )
        else:
            det_query = " or ".join(f"tag=='{x}'" for x in self.valid_tags)
        nondet_query = " or ".join(f"tag=='{x}'" for x in self.nondet_tags)
        if tag_col in self.lightcurve.columns:
            self.detections = self.lightcurve.query(det_query)
            self.non_detections = self.lightcurve.query(nondet_query)
        else:
            self.detections = self.lightcurve
            self.non_detections = None
        return

    def integrate_lightcurve_updates(
        self,
        updates: pd.DataFrame,
        column: str = "candid",
        nan_values: List[str] = (0,),
        timelike=False,
        verify_integrity=True,
        keep_updates=True,
    ):
        """
        combine updates into a lightcurve.

        Parameters
        ----------
        updates
            pd.DataFrame, the updates you want to include in your updated lightcurve.
        column
            the column to check for matches, and check that repeated rows don't happen


        """
        if column not in self.lightcurve.columns:
            raise ValueError(f"{column} not in both lightcurve columns")
        if column not in updates.columns:
            raise ValueError(f"{column} not in both updates columns")
        if timelike:
            updated_lightcurve = self.integrate_timelike(
                updates, column, keep_updated=keep_updates, nan_values=nan_values
            )
        else:
            updated_lightcurve = self.integrate_equality(
                updates,
                column,
                verify_integrity=verify_integrity,
                nan_values=nan_values,
            )
        self.add_lightcurve(updated_lightcurve)

    def integrate_timelike(self, updates, column, keep_updated=True):
        raise NotImplementedError

    def integrate_equality(
        self, updates: pd.DataFrame, column, verify_integrity=True, nan_values=(0,)
    ):
        if self.lightcurve is None:
            return updates
        updated_lightcurve = pd.concat([self.lightcurve, updates])

        if verify_integrity:
            extra_nan_vals_mask = updated_lightcurve[column].isin(nan_values)
            pd_nan_mask = updated_lightcurve[column].isna()
            finite_col_vals = updated_lightcurve[extra_nan_vals_mask & pd_nan_mask][
                column
            ]
            if not finite_col_vals.is_unique:
                repeated_mask = nonzero_candid["candid"].duplicated()
                repeated_candid = nonzero_candid["candid"][repeated_mask]
                err_msg = f"repeated candid:\n{set(repeated_candid)}"
                raise ValueError(err_msg)
        return updated_lightcurve


class Target:
    """
    TODO: docstring here!
    """

    default_base_score = 100.0
    # data_sources = ("alerce", "atlas", "fink", "tns")
    default_broker_priority = ("alerce", "fink", "lasair")

    def __init__(
        self,
        objectId: str,
        ra: float,
        dec: float,
        alerce_data: TargetData = None,
        atlas_data: TargetData = None,
        fink_data: TargetData = None,
        lasair_data: TargetData = None,
        tns_data: TargetData = None,
        yse_data: TargetData = None,
        base_score: float = None,
        target_of_opportunity: bool = False,
        broker_priority: tuple = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        # Basics
        self.objectId = objectId
        self.update_coordinates(ra, dec)
        self.base_score = base_score or self.default_base_score
        self.compiled_lightcurve = None
        self.broker_priority = broker_priority or self.default_broker_priority
        _check_broker_priority(self.broker_priority)

        # Target data
        self.alerce_data = alerce_data or TargetData()
        self.atlas_data = atlas_data or TargetData()
        self.fink_data = fink_data or TargetData()
        self.lasair_data = lasair_data or TargetData()
        self.tns_data = tns_data or TargetData()
        self.yse_data = yse_data or TargetData()

        # Observatory data
        self.observatory_information = {"no_observatory": None}

        # Scoring data
        self.models = {}
        self.models_tref = {}
        self.score_history = {"no_observatory": []}
        self.score_comments = {"no_observatory": None}
        self.reject_comments = {"no_observatory": None}
        self.rank_history = {"no_observatory": []}

        # Space to keep some plots
        self.latest_lc_fig = None
        self.latest_lc_fig_path = None
        self.latest_oc_figs = {}
        self.latest_oc_fig_paths = {}

        # Keep track of what's going on
        self.creation_time = t_ref
        self.target_of_opportunity = target_of_opportunity
        self.updated = False
        self.to_reject = False
        self.send_updates = False
        self.update_messages = []

    def __str__(self):
        s = f"{self.objectId}: ra={self.ra:.5f} dec={self.dec:.5f}\n"
        return s

    def update_coordinates(self, ra, dec):
        self.ra = ra
        self.dec = dec
        if ra is not None and dec is not None:
            self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            self.astroplan_target = FixedTarget(self.coord, self.objectId)  # for plots

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
            a callable function, which accepts three parameters, `target`, `observatory` and `t_ref`:
                - `target`: an instance of `Target` (ie, this class). `self` will be passed.
                - `observatory`: `astroplan.Observer`, or `None`
                - `t_ref`: an `astropy.time.Time`
            it should return one or three objects:
                - `score`: a float
                - `score_comments`: an iterable (list) of strings explaining the score.
                - `reject_comments`: an iterable (list) of strings explaing rejection, or `None`.

        """

        t_ref = t_ref or Time.now()
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None

        scoring_result = scoring_function(self, observatory)  # `self` is first arg!

        ##===== TODO: Check signature of function! =====##
        error_msg = (
            f"your function '{scoring_function.__name__}' should return "
            f"the score, and optionally two lists of strings "
            f"`score_comments` and `reject_comments`, not {scoring_result} "
            f"which is {type(scoring_result)}"
        )
        if isinstance(scoring_result, float) or isinstance(scoring_result, int):
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
        obs_name: str [optional]
            the name of an observatory that the system knows about
            if not provided defaults to `no_observatory`.
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

        # obs_score_history = self.score_history.get(obs_name, [])
        # if len(obs_score_history) == 0:
        #    result = (None, None)
        # else:
        #    result = obs_score_history[-1]

        obs_history = self.score_history.get(obs_name, [None])
        obs_history = obs_history or [None]
        result = obs_history[-1] or (None, None)

        if return_time:
            return result
        else:
            return result[0]

    def build_compiled_lightcurve(
        self,
        compile_function: Callable,
        t_ref: Time = None,
        lazy=True,
        broker_priority: list = None,
    ):
        t_ref = t_ref or Time.now()
        if lazy and not self.updated and self.compiled_lightcurve is not None:
            logger.debug(f"{self.objectId}: skip compile, not updated and lazy=True")
            return

        if compile_function is None:
            logger.warning("no light curve compiled.")
            return

        compiled_lightcurve = compile_function(self)
        if compiled_lightcurve is not None:
            compiled_lightcurve.sort_values("jd", inplace=True)
            compiled_lightcurve.query(f"jd < @t_ref.jd", inplace=True)
        self.compiled_lightcurve = compiled_lightcurve

    def build_model(self, modeling_function: Callable, t_ref: Time = None, lazy=True):
        if lazy and not self.updated:
            return

        models = {}
        models_tref = {}
        try:
            model = modeling_function(self)
        except Exception as e:
            logger.warn(
                f"error modeling {self.objectId} with {modeling_function.__name__}:"
            )
            tr = traceback.format_exc()
            print(tr)
            model = None
        key_name = modeling_function.__name__  # model.__class__.__name__
        models[key_name] = model
        models_tref[key_name] = t_ref
        self.models.update(models)
        self.models_tref.update(models_tref)
        return model

    def plot_lightcurve(
        self,
        lc_plotting_function: Callable = None,
        t_ref: Time = None,
        fig=None,
        figpath=None,
    ):
        t_ref = t_ref or Time.now()
        if lc_plotting_function is None:
            lc_plotting_function = default_plot_lightcurve
        lc_fig = lc_plotting_function(self, t_ref=t_ref, fig=fig)
        # self.latest_lc_fig = lc_fig
        if figpath is not None and lc_fig is not None:
            lc_fig.savefig(figpath)
            self.latest_lc_fig_path = figpath
        return lc_fig

    def plot_observing_chart(self, observatory, t_ref: Time = None, figpath=None):
        obs_name = getattr(observatory, "name", "no_observatory")
        t_ref = t_ref or Time.now()

        try:
            oc_fig = plot_observing_chart(observatory, self)
        except Exception as e:
            print(traceback.format_exc())
            oc_fig = None
        if oc_fig is not None and figpath is not None:
            oc_fig.savefig(figpath)
            self.latest_oc_fig_paths[obs_name] = figpath
        return oc_fig

    def reset_figures(self):
        # self.latest_lc_fig = None
        self.latest_lc_fig_path = None
        # self.latest_oc_figs = {}
        self.latest_oc_fig_paths = {}


def _check_broker_priority(broker_list):
    unknown_brokers = []
    for broker in broker_list:
        if broker not in Target.default_broker_priority:
            unknown_brokers.append(broker)
    if len(unknown_brokers) > 0:
        raise ValueError(
            f"unknown brokers {unknown_brokers}: choose from {Target.default_broker_priority}"
        )


lc_gs = plt.GridSpec(3, 4)
zscaler = ZScaleInterval()


def default_plot_lightcurve(
    target: Target, t_ref: Time = None, fig=None, forecast_days=10.0
) -> plt.Figure:
    t_ref = t_ref or Time.now()

    ##======== initialise figure
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(lc_gs[:, :-1])
    else:
        ax = fig.axes[0]

    if target.compiled_lightcurve is None:
        return None

    det_kwargs = dict(ls="none", marker="o")
    ulim_kwargs = dict(ls="none", marker="v", mfc="none")
    badqual_kwargs = dict(ls="none", marker="o", mfc="none")

    peak_mag_vals = []
    legend_handles = []

    lightcurve_plot_colours = {}

    for ii, (band, band_history) in enumerate(
        target.compiled_lightcurve.groupby("band")
    ):
        band_history.sort_values("jd")
        band_kwargs = dict(color=f"C{ii%8}")
        lightcurve_plot_colours[band] = f"C{ii%8}"

        scatter_handle = ax.errorbar(
            0, 0, yerr=0.1, label=band, **band_kwargs, **det_kwargs
        )
        legend_handles.append(scatter_handle)

        if "tag" in band_history:
            detections = band_history.query("tag=='valid'")
            ulimits = band_history.query("tag=='upperlim'")
            badqual = band_history.query("tag=='badquality'")

            Npoints = len(band_history)
            Ndet = len(detections)
            Nulim = len(ulimits)
            Nbq = len(badqual)

            if not (Ndet + Nulim + Nbq) == Npoints:
                msg = f"for band {band} len(det+ulimits+badqual)!=len(df)"
                msg = msg + f"({Ndet}+{Nulim}+{Nbq}!={Npoints})"
                logger.warning(msg)

            if len(ulimits) > 0:
                xdat = ulimits["jd"].values - t_ref.jd
                ydat = ulimits["diffmaglim"]
                ax.errorbar(xdat, ydat, **band_kwargs, **ulim_kwargs)

            if len(badqual) > 0:
                xdat = badqual["jd"].values - t_ref.jd
                ydat = badqual["mag"].values
                yerr = badqual["magerr"].values
                ax.errorbar(xdat, ydat, yerr=yerr, **band_kwargs, **badqual_kwargs)
        else:
            detections = band_history

        if any(np.isfinite(detections["mag"])):
            peak_mag_vals.append(np.nanmin(detections["mag"]))

        if len(detections) > 0:
            xdat = detections["jd"] - t_ref.jd
            ydat = detections["mag"]
            yerr = detections["magerr"]
            ax.errorbar(xdat, ydat, yerr=yerr, **band_kwargs, **det_kwargs)

    peak_mag_vals.append(16.0)
    y_bright = np.nanmin(peak_mag_vals)
    ax.set_ylim(22.0, y_bright)
    ax.axvline(t_ref.jd - t_ref.jd, color="k")

    legend = ax.legend(handles=legend_handles, loc=2)
    ax.add_artist(legend)

    title = f"{target.objectId}, ra={target.ra:.4f} dec={target.dec:.5f}"
    known_redshift = target.tns_data.parameters.get("Redshift", None)
    if known_redshift is not None:
        title = title + r" $z_{\rm TNS}=" + f"{known_redshift:.3f}" + "$"

    ax.text(
        0.5, 1.0, title, fontsize=14, ha="center", va="bottom", transform=ax.transAxes
    )

    date_str = t_ref.strftime("%d-%b-%y %H:%M")
    xlabel = f"Days before {date_str}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Difference magnitude")

    ##======== try sncosmo models
    model = target.models.get("sncosmo_model")
    if model is not None:
        t_start = target.compiled_lightcurve["jd"].min()
        t_end = t_ref.jd + forecast_days
        model_time_grid = np.arange(t_start, t_end, 0.1)
        for band, color in lightcurve_plot_colours.items():
            # model_mag = model.bandmag(band, "ab", model_time_grid)
            model_flux = model.bandflux(band, model_time_grid, zp=8.9, zpsys="ab")
            pos_mask = model_flux > 0.0
            model_mag = -2.5 * np.log10(model_flux[pos_mask]) + 8.9
            ax.plot(model_time_grid[pos_mask] - t_ref.jd, model_mag, color=color)

    ##======== add postage stamps
    cutouts = {}
    for broker in target.broker_priority:
        source_data = getattr(target, f"{broker}_data", None)
        if len(source_data.cutouts) == 0:
            continue
        cutouts = source_data.cutouts

    for ii, imtype in enumerate(["Science", "Template", "Difference"]):
        if len(fig.axes) == 4:
            im_ax = fig.axes[ii + 1]
        else:
            im_ax = fig.add_subplot(lc_gs[ii : ii + 1, -1:])

        im_ax.set_xticks([])
        im_ax.set_yticks([])

        imtext_kwargs = dict(
            rotation=90, transform=im_ax.transAxes, ha="left", va="center"
        )
        im_ax.text(1.02, 0.5, imtype, **imtext_kwargs)

        im = cutouts.get(imtype.lower(), None)
        if im is None:
            continue

        im_finite = im[np.isfinite(im)]

        vmin, vmax = zscaler.get_limits(im_finite.flatten())

        im_ax.imshow(im, vmin=vmin, vmax=vmax)

        xl_im = len(im.T)
        yl_im = len(im)
        # add pointers
        im_ax.plot([0.5 * xl_im, 0.5 * xl_im], [0.2 * yl_im, 0.4 * yl_im], color="r")
        im_ax.plot([0.2 * yl_im, 0.4 * yl_im], [0.5 * yl_im, 0.5 * yl_im], color="r")
    fig.tight_layout()

    comments = target.score_comments.get("no_observatory", [])
    if comments is not None:
        if len(comments) > 0:
            fig.subplots_adjust(bottom=0.3)
            text = "score comments:\n" + "\n".join(f"    {comm}" for comm in comments)
            fig.text(
                0.01, 0.01, text, ha="left", va="bottom", transform=fig.transFigure
            )
    return fig


def plot_observing_chart(
    observatory: Observer,
    target: Target = None,
    t_ref=None,
    fig=None,
    alt_ax=None,
    sky_ax=None,
):
    t_ref = t_ref or Time.now()

    if fig is None:
        fig = plt.figure(figsize=(6, 8))
    if alt_ax is None:
        alt_ax = fig.add_axes(211)
    if sky_ax is None:
        sky_ax = fig.add_axes(212, projection="polar")
    obs_name = getattr(observatory, "name", None)
    if observatory is None:
        return fig

    if target is not None:
        obs_info = target.observatory_information.get(obs_name, None)
        t_grid = obs_info.t_grid
        moon_altaz = obs_info.moon_altaz
        sun_altaz = obs_info.sun_altaz
        target_altaz = obs_info.target_altaz
    else:
        obs_info = None

    if obs_info is None:
        t_grid = t_ref + np.linspace(0, 24.0, 24 * 4) * u.hour
        moon_altaz = observatory.moon_altaz(t_grid)
        sun_altaz = observatory.sun_altaz(t_grid)
        obs_info = dict(t_grid=t_grid, moon_altaz=moon_altaz, sun_altaz=sun_altaz)

    alt_ax.plot(t_grid.mjd, moon_altaz.alt.deg, color="0.5", ls="--", label="moon")
    alt_ax.plot(t_grid.mjd, sun_altaz.alt.deg, color="0.5", ls=":", label="sun")
    alt_ax.set_ylim(-20, 90)
    alt_ax.axhline(0, color="k")
    alt_ax.set_ylabel("Altitude [deg]", fontsize=16)
    if target is not None:
        alt_ax.plot(t_grid.mjd, target_altaz.alt.deg, color="b", label="target")

        if all(target_altaz.alt < 30 * u.deg):
            bad_alt_kwargs = dict(
                color="red", rotation=45, ha="center", va="center", fontsize=18
            )
            text = f"target alt never >30 deg"
            alt_ax.text(0.5, 0.5, text, transform=alt_ax.transAxes, **bad_alt_kwargs)

    title = f"Observing from {obs_name}"
    title = title + f"\n starting at {t_ref.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    title_kwargs = dict(fontsize=14, ha="center", va="bottom")
    alt_ax.text(0.5, 1.0, title, transform=alt_ax.transAxes, **title_kwargs)

    # sun_alt = = (91 * u.deg - obs.altaz(time, target).alt) * (1/u.deg)

    sky_ax.set_rlim(bottom=90, top=0)
    sky_ax.plot(sun_altaz.az.rad, sun_altaz.alt.deg, color="0.5", ls=":")
    sky_ax.plot(moon_altaz.az.rad, moon_altaz.alt.deg, color="0.5", ls="--")
    if target is not None:
        sky_ax.plot(target_altaz.az.rad, target_altaz.alt.deg, color="r")

    return fig
