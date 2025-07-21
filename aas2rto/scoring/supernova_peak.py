import copy
import traceback
import time
from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto.target import Target
from aas2rto.target import DEFAULT_ZTF_BROKER_PRIORITY

logger = getLogger("supernova_peak")


def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


def gauss(x, A, mu, sig):
    return A * np.exp(-((x - mu) ** 2) / sig**2)


def calc_mag_factor(mag: float, zp: float = 18.5):
    """f=1 if mag=18.5, f~3 if mag=16.5, f=10 if mag=14.5,"""
    return 10 ** ((zp - mag) / 2)


def calc_timespan_factor(timespan: float, characteristic_timespan: float = 20.0):
    factor = 1.0
    if timespan > characteristic_timespan:
        factor = logistic(timespan, 1.0, -1, characteristic_timespan + 5.0)
        # +5 meanst that it starts to fall of ~ish at t=char_timespan
    return max(factor, 1e-3)


def calc_rising_fraction(band_det: pd.DataFrame, single_point_value: float = 0.5):
    if len(band_det) > 1:
        interval_rising = band_det["mag"].values[:-1] > band_det["mag"].values[1:]
        rising_fraction = sum(interval_rising) / len(interval_rising)
    else:
        rising_fraction = single_point_value
    return max(rising_fraction, 0.05)


def tuned_interest_function(x):
    return logistic(x, 10.0, -1.0 / 2.0, -6.0) + gauss(x, 4.0, 0.0, 1.0)


def peak_only_interest_function(x):
    return max(gauss(x, 30.0, 0.0, 2.0), 1e-2)


def calc_color_factor(gmag, rmag):
    gr_color = gmag - rmag
    color_factor = 1 + np.exp(-5.0 * gr_color)
    return color_factor


DEFAULT_SCORING_SOURCES = ["ztf", "yse"]
gal_center = SkyCoord(frame="galactic", l=0.0, b=0.0, unit="deg")


class SupernovaPeakScore:
    """

    Parameters
    ----------
    faint_limit
    min_timespan
    max_timespan
    charachteristic_timespan
    min_rising_fraction
    min_detections
    default_color_factor
    scoring_sources: tuple, default=(ztf, yse)
        which sources should we consider when computing eg. timespan
    """

    def __init__(
        self,
        faint_limit: float = 19.0,
        min_timespan: float = 1.0,
        max_timespan: float = 30.0,
        characteristic_timespan: float = 20.0,
        min_rising_fraction: float = 0.4,
        min_detections: int = 3,
        default_color_factor: float = 0.1,
        min_bulge_sep=10.0,
        broker_priority: tuple = DEFAULT_ZTF_BROKER_PRIORITY,
        use_compiled_lightcurve=True,
        scoring_sources: tuple = DEFAULT_SCORING_SOURCES,
    ):
        self.__name__ = "supernova_peak_score"  # self.__class__.__name__

        self.faint_limit = faint_limit
        self.min_timespan = min_timespan
        self.max_timespan = max_timespan
        self.characteristic_timespan = characteristic_timespan
        self.min_rising_fraction = min_rising_fraction
        self.default_color_factor = default_color_factor
        self.min_bulge_sep = min_bulge_sep
        self.min_detections = min_detections
        self.broker_priority = tuple(broker_priority)
        self.use_compiled_lightcurve = use_compiled_lightcurve
        self.scoring_sources = scoring_sources

    def __call__(self, target: Target, t_ref: Time) -> Tuple[float, List]:
        # t_ref = t_ref or Time.now()

        # to keep track
        scoring_comments = []

        exclude_comments = []
        reject_comments = []
        factors = {}
        reject = False
        exclude = False  # If true, don't reject, but not interesting right now.

        target_id = target.target_id

        ###===== Get the 'good' data (ZTF, PS1, etc.) =====###

        if target.compiled_lightcurve is None:
            return -1.0, ["no compiled lightcurve"]

        clc = target.compiled_lightcurve
        mask_list = []
        for source in self.scoring_sources:
            source_mask = (clc["source"] == source) & (clc["tag"] == "valid")
            mask_list.append(source_mask.values)

        detections_mask = sum(mask_list).astype(bool)
        detections = clc[detections_mask]

        if len(detections) == 0:
            return -1.0, ["no detections"]

        ###===== Make a quick check on the MJD data ===###
        if sum(detections["mjd"] > t_ref.mjd) > 0:
            mjd_max = detections["mjd"].max()
            mjd_warning = (
                f"{target_id}:\n    highest date ({mjd_max:.3f}) "
                f"later than t_ref={t_ref.mjd:.3f} ?!"
            )
            logger.warning(mjd_warning)
            detections = detections[detections["mjd"] < t_ref.mjd]

        ###===== Is it in the bulge? =======###
        gal_target = target.coord.galactic

        bulge_sep = target.coord.separation(gal_center)
        if bulge_sep.deg < self.min_bulge_sep:
            reject = True
            coord_string = f"(l,b)=({gal_target.l.deg:.1f},{gal_target.b.deg:.2f})"
            scoring_comments.append(
                f"REJECT: {coord_string} < {self.min_bulge_sep:.1f} from MW center"
            )

        ###===== Is it bright enough =======###
        last_mag = detections["mag"].values[-1]
        last_band = detections["band"].values[-1]
        mag_factor = calc_mag_factor(last_mag)
        scoring_comments.append(f"mag_factor={mag_factor:.2f} from mag={last_mag:.2f}")
        if last_mag > self.faint_limit:
            exclude = True
            comm = f"exclude: mag={last_mag:.1f}>{self.faint_limit} (faint lim)"
            exclude_comments.append(comm)
        factors["mag"] = mag_factor

        ###===== Is the target very old? ======###
        timespan = t_ref.mjd - detections["mjd"].min()
        factors["timespan"] = calc_timespan_factor(
            timespan, characteristic_timespan=self.characteristic_timespan
        )
        comm = f"timespan f={factors['timespan']:.2f} from timespan={timespan:.1f}d"
        scoring_comments.append(comm)
        if timespan > self.max_timespan:
            reject = True
            reject_comments.append(f"REJECT: target is {timespan:.2f} days old")

        ##==== or much too young!? ====##
        if timespan < self.min_timespan:
            exclude = True
            exclude_comment = (
                f"exclude: timespan {timespan:.2f}<{self.min_timespan:.2f} (min)"
            )
            exclude_comments.append(exclude_comment)

        ###===== Is the target still rising ======###
        unique_bands = detections["band"].unique()
        N_detections = {}

        f_rising_data = {}
        for band in unique_bands:
            band_detections = detections[detections["band"] == band]
            N_detections[band] = len(band_detections)

            rising_fraction_band = calc_rising_fraction(band_detections)
            # band_comm = f"f_rising={rising_fraction_band:.2f} for {len(band_detections)} band {band} obs"
            # scoring_comments.append(band_comm)

            f_rising_data[band] = rising_fraction_band
            factors[f"rising_fraction_{band}"] = rising_fraction_band
            if rising_fraction_band < self.min_rising_fraction:
                pass
                # reject = True
                # comm = f"REJECT: rising_fraction {band} is {rising_fraction_band:.2f} < {self.min_rising_fraction}"
                # reject_comments.append(comm)

        f_rising_comm = "f_rise:" + " ".join(
            f"{b}:{f:.2f}" for b, f in f_rising_data.items()
        )
        scoring_comments.append(f_rising_comm)

        ###===== How many observations =====###
        if len(detections) < self.min_detections:
            comm = f"exclude: {N_detections} detections insufficient"
            scoring_comments.append(comm)
            exclude = True

        ###===== Factors dependent on Model =====###

        model = target.models.get("sncosmo_salt", None)
        if model is not None:
            sncosmo_model = copy.deepcopy(model)
            # Get "result" if it exists, else empty dict. Then ask for "samples", else get None.

            ###===== Samples
            try:
                samples = getattr(model, "result", {}).get("samples", None)
            except Exception as e:
                # TODO: fix this block...
                samples = None

            if samples is not None:
                vparam_names = model.result.get("vparam_names")
                median_params = np.nanquantile(samples, q=0.5, axis=0)

                pdict = {k: v for k, v in zip(vparam_names, median_params)}
                sncosmo_model.update(pdict)

            ###===== Time from peak?
            t0 = sncosmo_model["t0"]
            peak_dt = t_ref.mjd - sncosmo_model["t0"]
            if not np.isfinite(peak_dt):
                interest_factor = 1.0
                msg = (
                    f"{target_id} interest_factor not finite:\n    "
                    f"dt={peak_dt:.2f}, mjd={t_ref.mjd:.2f}, t0={t0:.2f}"
                    f"{model.parameters}"
                )
                logger.warning(msg)
                scoring_comments.append(msg)
            else:
                interest_factor = peak_only_interest_function(peak_dt)

            factors["interest_factor"] = interest_factor
            interest_comment = (
                f"interest {interest_factor:.2f} from peak_dt={peak_dt:+.2f}d"
            )
            scoring_comments.append(interest_comment)

            if peak_dt > self.max_timespan:
                reject = True
                reject_comments.append(f"REJECT: too far past peak ({peak_dt:+.1f}d)")

            ###===== Blue colour?

            # ztfg_mag = sncosmo_model.bandmag("ztfg", "ab", t_ref.mjd)
            # ztfr_mag = sncosmo_model.bandmag("ztfr", "ab", t_ref.mjd)

            # if N_detections.get(1, 0) == 0 or N_detections.get(2, 0) == 0:
            #     color_factor = self.default_color_factor
            #     scoring_comments.append(f"color set as {color_factor:.1} due to N_det")
            # else:
            #     color_factor = calc_color_factor(ztfg_mag, ztfr_mag)
            # if not np.isfinite(color_factor):
            #     color_factor = self.default_color_factor
            #     color_comment = f"infinite color factor set as {color_factor:1f} (g={ztfg_mag}, r={ztfr_mag})"
            #     scoring_comments.append(color_comment)
            # else:
            #     scoring_comments.append(
            #         f"color factor={color_factor:.2f} from model g-r={ztfg_mag-ztfr_mag:.2f}"
            #     )
            # factors["color_factor"] = color_factor

            ###===== chisq

            try:
                model_result = getattr(model, "result", {})
            except Exception as e:
                model_result = None

            if model_result is not None:
                chisq = model_result.get("chisq", np.nan)
                ndof = model_result.get("ndof", np.nan)
                if (not np.isfinite(chisq)) or (not np.isfinite(ndof)):
                    scoring_comments.append(f"chisq={chisq:.2f} and ndof={ndof}")
                else:
                    try:
                        chisq_nu = chisq / ndof
                        msg = f"model chisq={chisq:.3f} with ndof={ndof}"
                        scoring_comments.append(msg)
                    except ZeroDivisionError as e:
                        scoring_comments.append(f"model has ndof=0")
                    # TODO how to use chisq_nu properly?

        scoring_factors = np.array(list(factors.values()))
        if not all(scoring_factors > 0):
            print(detections)
            neg_factors = "REJECT:\n" + "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not v > 0
            )
            reject_comments.append(neg_factors)
            logger.warning(f"{target_id} has -ve/inf/nan factors:\n{neg_factors}")
            exclude = True

        combined_factors = np.prod(scoring_factors)
        final_score = target.base_score * combined_factors

        # scoring_str = "\n".join(f"    {k}={v:.3f}" for k, v in factors.items())
        scoring_str = "\n".join(f"   {comm}" for comm in scoring_comments)
        logger.debug(f"{target_id} has factors:\n {scoring_str}")

        if exclude:
            final_score = -1.0
        if reject:
            reject_str = "\n".join(f"    {comm}" for comm in reject_comments)
            logger.debug(f"{target_id}:\n{reject_str}")
            final_score = -np.inf

        # nice if the exclude and reject comments are at the end.
        scoring_comments.extend(exclude_comments)
        scoring_comments.extend(reject_comments)

        return final_score, scoring_comments
