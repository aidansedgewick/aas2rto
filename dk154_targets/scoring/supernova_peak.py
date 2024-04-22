import copy
import traceback
from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

from astropy.time import Time

from astroplan import Observer

from dk154_targets import Target
from dk154_targets.target import DEFAULT_ZTF_BROKER_PRIORITY

logger = getLogger("supernova_peak")


def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


def gauss(x, A, mu, sig):
    return A * np.exp(-((x - mu) ** 2) / sig**2)


def calc_mag_factor(mag: float, zp=18.5):
    """f=1 if mag=18.5, f~3 if mag=16.5, f=10 if mag=14.5,"""
    return 10 ** ((zp - mag) / 2)


def calc_timespan_factor(timespan: float, characteristic_timespan=20.0):
    factor = 1.0
    if timespan > characteristic_timespan:
        factor = logistic(timespan, 1.0, -1, characteristic_timespan + 5.0)
        # +5 meanst that it starts to fall of ~ish
    return max(factor, 1e-3)


def calc_rising_fraction(band_det: pd.DataFrame, single_point_value=0.5):
    if len(band_det) > 1:
        interval_rising = band_det["magpsf"].values[:-1] > band_det["magpsf"].values[1:]
        rising_fraction = sum(interval_rising) / len(interval_rising)
    else:
        rising_fraction = single_point_value
    return max(rising_fraction, 0.05)


def tuned_interest_function(x):
    return logistic(x, 10.0, -1.0 / 2.0, -6.0) + gauss(x, 4.0, 0.0, 1.0)


def peak_only_interest_function(x):
    return max(gauss(x, 10.0, 0.0, 2.0), 1e-2)


def calc_color_factor(gmag, rmag):
    gr_color = gmag - rmag
    color_factor = 1 + np.exp(-5.0 * gr_color)
    return color_factor


def calc_visibility_factor(alt_grid, min_alt=30.0, ref_alt=45.0):
    alt_above_minimum = np.maximum(alt_grid, min_alt) - min_alt
    integral = np.trapz(alt_above_minimum)

    if min_alt > ref_alt:
        raise ValueError(f"norm_alt > min_altitude ({ref_alt} > {min_alt})")
    norm = len(alt_grid) * (ref_alt - min_alt)

    return 1.0 / (integral / norm)


def calc_altitude_factor(alt):
    """
    This is just sin(alt), with alt in deg.
    motivated by X~sec(90-alt), where X is airmass.
    promote high alt, so low airmass --> 1/X ~ sin(alt)
    """
    return np.sin(alt * np.pi / 180.0)


class SupernovaPeakScore:
    def __init__(
        self,
        faint_limit: float = 19.0,
        min_timespan: float = 1.0,
        max_timespan: float = 20.0,
        min_rising_fraction: float = 0.4,
        min_altitude: float = 30.0,
        ref_altitude: float = 90.0,
        exclude_daytime: bool = False,
        visibility_method: str = "visibility",
        min_ztf_detections: int = 3,
        default_color_factor: float = 0.1,
        broker_priority: tuple = None,
        **kwargs,
    ):
        self.__name__ = "supernova_peak_score"  # self.__class__.__name__

        self.faint_limit = faint_limit
        self.min_timespan = min_timespan
        self.max_timespan = max_timespan
        self.min_rising_fraction = min_rising_fraction
        self.min_altitude = min_altitude
        self.ref_altitude = ref_altitude
        self.exclude_daytime = exclude_daytime
        self.visibility_method = visibility_method
        self.default_color_factor = default_color_factor
        self.min_ztf_detections = min_ztf_detections
        self.broker_priority = broker_priority or DEFAULT_ZTF_BROKER_PRIORITY

        expected_vis_methods = ["visibility", "altitude"]
        if self.visibility_method not in expected_vis_methods:
            msg = (
                f"visibility_factor should be one of {expected_vis_methods},\n    "
                f"not {visibility_method}"
            )
            raise ValueError(msg)

    def __call__(
        self, target: Target, observatory: Observer, t_ref: Time
    ) -> Tuple[float, List, List]:
        # t_ref = t_ref or Time.now()

        # to keep track
        scoring_comments = []
        reject_comments = []
        factors = {}
        reject = False
        exclude = False  # If true, don't reject, but not interesting right now.

        objectId = target.objectId

        ztf_source = None
        for source in self.broker_priority:
            ztf_source = target.target_data.get(source, None)
            if ztf_source is None:
                continue
            if ztf_source.lightcurve is None:
                ztf_source = None
                continue
            break

        if ztf_source is None:
            scoring_comments.append(f"none of {self.broker_priority} available")
            return -1.0, scoring_comments, reject_comments
        ztf_detections = ztf_source.detections

        if any(ztf_detections["mjd"] > t_ref.mjd):
            mjd_max = ztf_detections["mjd"].max()
            logger.warning(
                f"{objectId}:\n    highest date ({mjd_max}) later than t_ref={t_ref.mjd} ?!"
            )

        ###===== Is it bright enough =======###
        ztf_detections = ztf_detections[ztf_detections["jd"] < t_ref.jd]
        last_mag = ztf_detections["magpsf"].values[-1]

        last_band = ztf_detections["fid"].values[-1]
        mag_factor = calc_mag_factor(last_mag)
        scoring_comments.append(f"mag_factor={mag_factor:.2f} from mag={last_mag:.2f}")
        if last_mag > self.faint_limit:
            exclude = True
            scoring_comments.append(
                f"latest mag {last_mag:.1f} too faint (>{self.faint_limit}): exclude from ranking"
            )
        factors["mag"] = mag_factor

        ###===== Is the target very old? ======###
        timespan = t_ref.mjd - ztf_detections["mjd"].min()
        factors["timespan"] = calc_timespan_factor(
            timespan, characteristic_timespan=self.max_timespan
        )
        timespan_comment = (
            f"timespan f={factors['timespan']:.2f} from timspan={timespan:.1f}d"
        )
        scoring_comments.append(timespan_comment)
        if timespan > self.max_timespan:
            reject = True
            reject_comments.append(f"target is {timespan:.2f} days old")

        ##==== or much too young!? ====##
        if timespan < self.min_timespan:
            exclude = True
            exclude_comment = f"timespan less than min={self.min_timespan}"
            scoring_comments.append(exclude_comment)

        ###===== Is the target still rising ======###
        for fid, fid_history in ztf_detections.groupby("fid"):
            fid_history.sort_values("mjd", inplace=True)
            rising_fraction_fid = calc_rising_fraction(fid_history)
            scoring_comments.append(
                f"f_rising={rising_fraction_fid:.2f} for {len(fid_history)} band {fid} obs"
            )
            factors[f"rising_fraction_{fid}"] = rising_fraction_fid
            if rising_fraction_fid < self.min_rising_fraction:
                reject = True
                reject_comments.append(
                    f"rising_fraction {fid} is {rising_fraction_fid:.2f} < {self.min_rising_fraction}"
                )

        ###===== How many observations =====###
        N_detections = {}
        for fid, fid_history in ztf_detections.groupby("fid"):
            N_detections[fid] = len(fid_history)
        if N_detections.get(1, 0) < 2 and N_detections.get(2, 0) < 2:
            scoring_comments.append(
                f"exclude as detections {N_detections} insufficient"
            )
            exclude = True
        if len(ztf_detections) < self.min_ztf_detections:
            scoring_comments.append(
                f"exclude as detections {N_detections} insufficient"
            )
            exclude = True

        model = target.models.get("sncosmo_salt", None)
        sncosmo_model = copy.deepcopy(model)
        if sncosmo_model is not None:
            # Get "result" if it exists, else empty dict. Then ask for "samples", else get None.
            samples = getattr(model, "result", {}).get("samples", None)
            if samples is not None:
                vparam_names = model.result.get("vparam_names")
                median_params = np.nanquantile(samples, q=0.5, axis=0)

                pdict = {k: v for k, v in zip(vparam_names, median_params)}
                sncosmo_model.update(pdict)

            ###===== Time from peak?
            t0 = sncosmo_model["t0"]
            peak_dt = t_ref.mjd - sncosmo_model["t0"]
            interest_factor = peak_only_interest_function(peak_dt)
            if not np.isfinite(interest_factor):
                interest_factor = 1.0
                msg = (
                    f"{objectId} interest_factor not finite:\n    "
                    f"dt={peak_dt:.2f}, mjd={t_ref.mjd:.2f}, t0={t0:.2f}"
                )
                logger.warning(msg)
                scoring_comments.append(msg)
            factors["interest_factor"] = interest_factor
            interest_comment = (
                f"interest {interest_factor:.2f} from peak_dt={peak_dt:.2f}d"
            )
            scoring_comments.append(interest_comment)

            if peak_dt > self.max_timespan:
                reject = True
                reject_comments.append(f"too far past peak {peak_dt:.1f}")

            ###===== Blue colour?
            ztfg_mag = sncosmo_model.bandmag("ztfg", "ab", t_ref.jd)
            ztfr_mag = sncosmo_model.bandmag("ztfr", "ab", t_ref.jd)

            if N_detections.get(1, 0) == 0 or N_detections.get(2, 0) == 0:
                color_factor = self.default_color_factor
                scoring_comments.append(f"color set as {color_factor:.1} due to N_det")
            else:
                color_factor = calc_color_factor(ztfg_mag, ztfr_mag)
            if not np.isfinite(color_factor):
                color_factor = self.default_color_factor
                color_comment = f"infinite color factor set as {color_factor:1f} (g={ztfg_mag}, r={ztfr_mag})"
                scoring_comments.append(color_comment)
            else:
                scoring_comments.append(
                    f"color factor={color_factor:.2f} from model g-r={ztfg_mag-ztfr_mag:.2f}"
                )
            factors["color_factor"] = color_factor

            ###===== chisq
            model_result = getattr(model, "result", {})
            chisq = model_result.get("chisq", np.nan)
            ndof = model_result.get("ndof", np.nan)
            if (not np.isfinite(chisq)) or (not np.isfinite(ndof)):
                scoring_comments.append(f"chisq={chisq:.2f} and ndof={ndof}")
            else:
                chisq_nu = chisq / ndof
                scoring_comments.append(f"model chisq={chisq:.3f} with ndof={ndof}")
                # TODO how to use chisq_nu properly?

        if observatory is not None:
            min_alt = self.min_altitude

            obs_name = getattr(observatory, "name", None)
            if obs_name is None:
                raise ValueError(f"observatory {observatory} has no name!!")
            obs_info = target.observatory_info.get(obs_name, None)
            if obs_info is None:
                logger.warning(f"obs_info=None for {obs_name} {objectId}")
                target_altaz = None
                logger.warning("calc curr_sun_alt. this is slow!")
                curr_sun_alt = observatory.sun_altaz(t_ref).alt.deg

            else:
                sunset = obs_info.sunset
                sunrise = obs_info.sunrise
                target_altaz = obs_info.target_altaz
                curr_sun_alt = obs_info.sun_altaz.alt.deg[0]

                if obs_info.sunset is None or obs_info.sunrise is None:
                    scoring_comments.append(f"sun never sets at this observatory")
                    exclude = True

            if self.exclude_daytime and curr_sun_alt > -18.0:
                scoring_comments.append(f"exclude as sun up.")
                exclude = True

            vis_method = self.visibility_method  # so we can force change it if we need.
            if target_altaz is not None:
                t_grid = obs_info.t_grid
                night_mask = (sunset.jd < t_grid.jd) & (t_grid.jd < sunrise.jd)
                night_alt = target_altaz.alt.deg[night_mask]
                curr_alt = target_altaz.alt.deg[0]

                if all(night_alt < min_alt):
                    exclude = True
            else:
                if vis_method != "altitude":
                    msg = (
                        f"{objectId} obs_info has no target alt_az"
                        "force vis_method='altitude'"
                    )
                    logger.warning(msg)
                vis_method = "altitude"
                night_alt = None
                curr_alt = observatory.altaz(t_ref, target.coord).alt.deg

            if curr_alt < min_alt:
                exclude = True

            if not exclude:
                if vis_method == "altitude":
                    vis_factor = calc_altitude_factor(curr_alt)
                    score_comm = (
                        f"vis_factor={vis_factor:.2f} from alt={curr_alt:.2f}deg"
                    )
                elif vis_method == "visibility":
                    vis_factor = calc_visibility_factor(
                        night_alt, min_alt=min_alt, ref_alt=self.ref_altitude
                    )
                else:
                    unexp_comm = f"unexpected method {vis_method}: vis_factor = 1.0"
                    scoring_comments.append(unexp_comm)
                    logger.warning(unexp_comm)
                    vis_factor = 1.0
                vis_comm = f"vis_factor={vis_factor:.2f} from alt={curr_alt:.2f}deg, method='{vis_method}'"
                scoring_comments.append(vis_comm)
                factors["vis_factor"] = vis_factor

        scoring_factors = np.array(list(factors.values()))
        if not all(scoring_factors > 0):
            neg_factors = "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not v > 0
            )
            reject_comments.append(neg_factors)
            logger.warning(f"{objectId} has negative factors:\n{neg_factors}")
            reject = True

        combined_factors = np.prod(scoring_factors)
        final_score = target.base_score * combined_factors

        # scoring_str = "\n".join(f"    {k}={v:.3f}" for k, v in factors.items())
        scoring_str = "\n".join(f"   {comm}" for comm in scoring_comments)
        logger.debug(f"{objectId} has factors:\n {scoring_str}")

        if exclude:
            final_score = -1.0
        if reject:
            reject_str = "\n".join(f"    {comm}" for comm in reject_comments)
            logger.debug(f"{objectId}")
            final_score = -np.inf
        return final_score, scoring_comments, reject_comments
