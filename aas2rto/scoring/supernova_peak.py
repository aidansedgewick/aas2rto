import copy
import traceback
import time
from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

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
        interval_rising = band_det["magpsf"].values[:-1] > band_det["magpsf"].values[1:]
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


class SupernovaPeakScore:
    def __init__(
        self,
        faint_limit: float = 19.0,
        min_timespan: float = 1.0,
        max_timespan: float = 25.0,
        characteristic_timespan: float = 20.0,
        min_rising_fraction: float = 0.4,
        min_ztf_detections: int = 3,
        default_color_factor: float = 0.1,
        broker_priority: tuple = DEFAULT_ZTF_BROKER_PRIORITY,
        use_compiled_lightcurve=False,
    ):
        self.__name__ = "supernova_peak_score"  # self.__class__.__name__

        self.faint_limit = faint_limit
        self.min_timespan = min_timespan
        self.max_timespan = max_timespan
        self.characteristic_timespan = characteristic_timespan
        self.min_rising_fraction = min_rising_fraction
        self.default_color_factor = default_color_factor
        self.min_ztf_detections = min_ztf_detections
        self.broker_priority = tuple(broker_priority)
        self.use_compiled_lightcurve = use_compiled_lightcurve

    def __call__(self, target: Target, t_ref: Time) -> Tuple[float, List, List]:
        # t_ref = t_ref or Time.now()

        # to keep track
        scoring_comments = []
        reject_comments = []
        factors = {}
        reject = False
        exclude = False  # If true, don't reject, but not interesting right now.

        objectId = target.objectId

        ###===== Get the ZTF data (fink, lasair, etc.) =====###

        if self.use_compiled_lightcurve:
            clc = target.compiled_lightcurve
            ztf_det_mask = (clc["source"] == "ztf") & (clc["tag"] == "valid")
            ztf_detections = clc[ztf_det_mask]
        else:
            ztf_source = None
            # ztf_source_name = None
            for source in self.broker_priority:
                ztf_source = target.target_data.get(source, None)
                if ztf_source is None:
                    continue
                if ztf_source.lightcurve is None:
                    # If eg. FINK has a fink "TargetData" obj, but no lightcurve...
                    msg = f"{objectId} has `{source}` TargetData, but no lightcurve!"
                    logger.warning(msg)
                    ztf_source = None  # ...it's not useful, so skip anyway.
                    continue
                source_name = source
                break

            if ztf_source is None:
                msg = f"{objectId}: none of {self.broker_priority} available"
                scoring_comments.append(msg)
                logger.warning(msg)
                scoring_comments.extend(reject_comments)
                return -1.0, scoring_comments
            ztf_detections = ztf_source.detections

        ###===== Make a quick check on the MJD data... ===###
        if sum(ztf_detections["mjd"] > t_ref.mjd) > 0:
            mjd_max = ztf_detections["mjd"].max()
            mjd_warning = (
                f"{objectId}:\n    highest date ({mjd_max:.3f}) "
                f"later than t_ref={t_ref.mjd:.3f} ?!"
            )
            logger.warning(mjd_warning)
            ztf_detections = ztf_detections[ztf_detections["mjd"] < t_ref.mjd]

        ###===== Is it bright enough =======###
        last_mag = ztf_detections["magpsf"].values[-1]
        last_band = ztf_detections["fid"].values[-1]
        mag_factor = calc_mag_factor(last_mag)
        scoring_comments.append(f"mag_factor={mag_factor:.2f} from mag={last_mag:.2f}")
        if last_mag > self.faint_limit:
            exclude = True
            scoring_comments.append(
                f"exclude: mag={last_mag:.1f}>{self.faint_limit} (faint lim)"
            )
        factors["mag"] = mag_factor

        ###===== Is the target very old? ======###
        timespan = t_ref.mjd - ztf_detections["mjd"].min()
        factors["timespan"] = calc_timespan_factor(
            timespan, characteristic_timespan=self.characteristic_timespan
        )
        comm = f"timespan f={factors['timespan']:.2f} from timespan={timespan:.1f}d"
        scoring_comments.append(comm)
        if timespan > self.max_timespan:
            reject = True
            reject_comments.append(f"target is {timespan:.2f} days old")

        ##==== or much too young!? ====##
        if timespan < self.min_timespan:
            exclude = True
            exclude_comment = f"exclude: timespan<{self.min_timespan} (min)"
            scoring_comments.append(exclude_comment)

        ###===== Is the target still rising ======###
        unique_fid = ztf_detections["fid"].unique()
        N_detections = {}
        for fid in unique_fid:
            fid_detections = ztf_detections[ztf_detections["fid"] == fid]
            N_detections[fid] = len(fid_detections)

            rising_fraction_fid = calc_rising_fraction(fid_detections)
            fid_comm = f"f_rising={rising_fraction_fid:.2f} for {len(fid_detections)} band {fid} obs"
            scoring_comments.append(fid_comm)
            factors[f"rising_fraction_{fid}"] = rising_fraction_fid
            if rising_fraction_fid < self.min_rising_fraction:
                reject = True
                comm = f"rising_fraction {fid} is {rising_fraction_fid:.2f} < {self.min_rising_fraction}"
                reject_comments.append(comm)

        ###===== How many observations =====###
        if len(ztf_detections) < self.min_ztf_detections:
            comm = f"exclude as detections {N_detections} insufficient"
            scoring_comments.append(comm)
            exclude = True

        ###===== Factors dependent on Model =====###

        model = target.models.get("sncosmo_salt", None)
        if model is not None:
            sncosmo_model = copy.deepcopy(model)
            # Get "result" if it exists, else empty dict. Then ask for "samples", else get None.

            ###===== Samples
            samples = getattr(model, "result", {}).get("samples", None)
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
                    f"{objectId} interest_factor not finite:\n    "
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

            model_result = getattr(model, "result", {})
            chisq = model_result.get("chisq", np.nan)
            ndof = model_result.get("ndof", np.nan)
            if (not np.isfinite(chisq)) or (not np.isfinite(ndof)):
                scoring_comments.append(f"chisq={chisq:.2f} and ndof={ndof}")
            else:
                chisq_nu = chisq / ndof
                scoring_comments.append(f"model chisq={chisq:.3f} with ndof={ndof}")
                # TODO how to use chisq_nu properly?

        scoring_factors = np.array(list(factors.values()))
        if not all(scoring_factors > 0):
            neg_factors = "REJECT:\n" + "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not v > 0
            )
            reject_comments.append(neg_factors)
            logger.warning(f"{objectId} has negative factors:\n{neg_factors}")
            exclude = True

        combined_factors = np.prod(scoring_factors)
        final_score = target.base_score * combined_factors

        # scoring_str = "\n".join(f"    {k}={v:.3f}" for k, v in factors.items())
        scoring_str = "\n".join(f"   {comm}" for comm in scoring_comments)
        logger.debug(f"{objectId} has factors:\n {scoring_str}")

        if exclude:
            final_score = -1.0
        if reject:
            reject_str = "\n".join(f"    {comm}" for comm in reject_comments)
            logger.debug(f"{objectId}:\n{reject_str}")
            final_score = -np.inf

        scoring_comments.extend(reject_comments)

        return final_score, scoring_comments
