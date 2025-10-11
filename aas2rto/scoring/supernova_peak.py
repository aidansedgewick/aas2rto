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


def calc_chisq_factor(chisq_nu):
    if chisq_nu < 5 and (1.0 / chisq_nu < 5):
        chisq_factor = 1.0
    elif chisq_nu < 10 and (1.0 / chisq_nu < 10.0):
        chisq_factor = 0.5
    else:
        chisq_factor = 0.1
    return chisq_factor


def calc_cv_prob_penalty(detections, ulim, model, cv_cand_detections=3, mag_thresh=1.0):
    """
    For upperlimits data before the model peak:

    if the detection limit is significantly below the model mag, penalise that data point
    """

    t0 = model["t0"]

    det_pre_t0 = detections[detections["mjd"] < t0]

    if len(ulim) == 0 or "diffmaglim" not in ulim.columns:
        return 1.0, [f"no CV-penalty (no 'diffmaglim' data)"]

    if len(det_pre_t0) > cv_cand_detections:
        return 1.0, [f"no CV penalty ({len(det_pre_t0)} det pre-t0)"]

    penalties = []
    # comments = []

    penalty_strings = []
    for band, band_ulim in ulim.groupby("band"):

        if band not in detections["band"]:
            continue

        t0_mask = band_ulim["mjd"] > t0 - 15.0  # anything after t0-15
        pre_t0_ulim = band_ulim[t0_mask]
        try:
            modelmag = model.bandmag(band, "ab", pre_t0_ulim["mjd"].values)
        except Exception as e:
            continue

        diff = pre_t0_ulim["diffmaglim"] - modelmag  # diff > 0? ulim fainter than model
        large_diff_mask = diff > mag_thresh

        # meas_penalties = np.minimum(1.0, np.exp(mag_thresh - diff))
        meas_penalties = np.exp(mag_thresh - diff[large_diff_mask])
        band_penalty = max(np.prod(meas_penalties), 0.01)
        penalties.append(band_penalty)
        if band_penalty < 0.9999:
            penalty_strings.append(f"{band}={band_penalty:.3f}")
            # comments.append(f"CV penalty {band_penalty:.3f} for {band}")

    cv_penalty = np.prod(penalties)

    comments = []
    if cv_penalty < 1.0:
        comments.append(f"CV penalty: {cv_penalty:.3e}")
        n_items = 3
        for ii in range(0, len(penalty_strings), n_items):
            bands_comm = "CV-pen: " + " ".join(penalty_strings[ii : ii + n_items])
            comments.append(bands_comm)
    else:
        comments.append("no CV penalty")

    return cv_penalty, comments


def tuned_interest_function(x):
    return logistic(x, 10.0, -1.0 / 2.0, -6.0) + gauss(x, 4.0, 0.0, 1.0)


def peak_only_interest_function(x):
    return max(gauss(x, 30.0, 0.0, 2.0), 1e-2)


def calc_color_factor(gmag, rmag):
    gr_color = gmag - rmag
    color_factor = 1 + np.exp(-5.0 * gr_color)
    return color_factor


DEFAULT_SCORING_SOURCES = ["ztf", "yse", "atlas"]
DEFAULT_SCORING_BANDS = "ztfg ztfr ztfi".split() + "ps1::g ps1::r ps1::i ps1::z".split()
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
        min_lc_length: float = 0.75,
        default_color_factor: float = 0.1,
        max_chisq_nu: float = 10.0,
        min_bulge_sep=15.0,
        min_gal_lat=5.0,
        # broker_priority: tuple = DEFAULT_ZTF_BROKER_PRIORITY,
        use_compiled_lightcurve=True,
        scoring_sources: tuple = DEFAULT_SCORING_SOURCES,
        # scoring_bands: tuple = DEFAULT_SCORING_BANDS,
    ):
        self.__name__ = "supernova_peak_score"  # self.__class__.__name__

        self.faint_limit = faint_limit
        self.min_timespan = min_timespan
        self.max_timespan = max_timespan
        self.characteristic_timespan = characteristic_timespan
        self.min_rising_fraction = min_rising_fraction
        self.min_detections = min_detections
        self.min_lc_length = min_lc_length
        self.default_color_factor = default_color_factor
        self.max_chisq_nu = max_chisq_nu
        self.min_bulge_sep = min_bulge_sep
        self.min_gal_lat = min_gal_lat
        # self.broker_priority = tuple(broker_priority)
        self.use_compiled_lightcurve = use_compiled_lightcurve
        self.scoring_sources = scoring_sources
        # self.scoring_bands = scoring_bands

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
        det_mask_list = []
        ulim_mask_list = []
        for source in self.scoring_sources:
            source_det_mask = (clc["source"] == source) & (clc["tag"] == "valid")
            det_mask_list.append(source_det_mask.values)

            source_ulim_mask = (clc["source"] == source) & (clc["tag"] == "upperlim")
            ulim_mask_list.append(source_ulim_mask.values)

        detections_mask = sum(det_mask_list).astype(bool)
        detections = clc[detections_mask]

        ulim_mask = sum(ulim_mask_list).astype(bool)
        upperlimits = clc[ulim_mask]

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

        ###===== Is it in the bulge or disc? =======###
        gal_target = target.coord.galactic

        bulge_sep = target.coord.separation(gal_center)
        if bulge_sep.deg < self.min_bulge_sep:
            reject = True
            coord_string = f"(l,b)=({gal_target.l.deg:.1f},{gal_target.b.deg:.2f})"
            scoring_comments.append(
                f"REJECT: {coord_string} < {self.min_bulge_sep:.1f} from MW center"
            )
        if abs(gal_target.b.deg) < self.min_gal_lat:
            reject = True
            coord_string = f"(l,b)=({gal_target.l.deg:.1f},{gal_target.b.deg:.2f})"
            scoring_comments.append(
                f"REJECT: {coord_string}: abs(b) < {self.min_gal_lat:.1f} - in the disc!"
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

        ##===== or much too young!? =====##
        if timespan < self.min_timespan:
            exclude = True
            comm = f"exclude: timespan {timespan:.2f}<{self.min_timespan:.2f} (min)"
            exclude_comments.append(comm)

        ##===== Is the target only a single night? =====##
        lc_length = detections["mjd"].max() - detections["mjd"].min()
        if lc_length < self.min_lc_length:
            exclude = True
            comm = f"exclude as lc length {lc_length:.2f} < {self.min_lc_length} (min)"
            exclude_comments.append(comm)

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

            if band_detections["source"].iloc[0] not in self.scoring_sources:
                continue

            f_rising_data[band] = rising_fraction_band
            factors[f"rising_fraction_{band}"] = rising_fraction_band
            if rising_fraction_band < self.min_rising_fraction:
                pass
                # reject = True
                # comm = f"REJECT: rising_fraction {band} is {rising_fraction_band:.2f} < {self.min_rising_fraction}"
                # reject_comments.append(comm)

        f_rising_strings = [f"{b}={f:.2f}" for b, f in f_rising_data.items()]

        n_items = 3
        for ii in range(0, len(f_rising_strings), n_items):
            f_rising_comm = "f_rise: " + " ".join(f_rising_strings[ii : ii + n_items])
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

            ###===== chisq
            try:
                model_result = getattr(model, "result", {})
            except Exception as e:
                model_result = None

            chisq_nu = None
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
                        scoring_comments.append(f"model has ndof={ndof} ?")

            if isinstance(chisq_nu, float):
                if chisq_nu > 0.0:
                    f_chisq = calc_chisq_factor(chisq_nu)
                    factors["chisq_factor"] = f_chisq
                    chisq_comm = f"chisq factor {f_chisq:.2f} (chisq_nu={chisq_nu:.3f})"
                else:
                    chisq_comm = f"no factor based on chisq_nu={chisq_nu:.3f}"
            else:
                chisq_comm = f"no f_chisq: non-float chisq_nu={chisq_nu})"
            scoring_comments.append(chisq_comm)

            ###===== Time from peak?
            t0 = sncosmo_model["t0"]
            peak_dt = t_ref.mjd - sncosmo_model["t0"]
            if not np.isfinite(peak_dt):
                f_interest = 1.0
                msg = (
                    f"{target_id} interest_factor not finite:\n    "
                    f"dt={peak_dt:.2f}, mjd={t_ref.mjd:.2f}, t0={t0:.2f}"
                    f"{model.parameters}"
                )
                logger.warning(msg)
                scoring_comments.append(msg)
            else:
                f_interest = peak_only_interest_function(peak_dt)

            if isinstance(chisq_nu, float):
                if (0.0 < chisq_nu) and chisq_nu < self.max_chisq_nu:
                    factors["interest_factor"] = f_interest
                    comm = f"interest {f_interest:.2f} from peak_dt={peak_dt:+.2f}d"
                    scoring_comments.append(comm)
                else:
                    comm = f"ignore interest {f_interest:.2f}/peak_dt={peak_dt:+.2f}d because chisq_nu={chisq_nu:.3f}"

                if peak_dt > self.max_timespan:
                    reject = True
                    comm = f"REJECT: too far past peak ({peak_dt:+.1f}d)"
                    reject_comments.append(comm)

            else:
                comm = f"ignore interest as non-float chisq_nu {chisq_nu}"
                scoring_comments.append(comm)

            ###===== Penalty for likely CVs?
            cv_penalty, cv_comms = calc_cv_prob_penalty(detections, upperlimits, model)
            factors["cv_penalty"] = cv_penalty
            scoring_comments.extend(cv_comms)

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

        scoring_factors = np.array(list(factors.values()))
        if not all(scoring_factors > 0):
            print(detections)
            neg_factors = "EXCLUDE:\n" + "\n".join(
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
