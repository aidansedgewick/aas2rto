from logging import getLogger

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto import Target

logger = getLogger("kn_score")

gal_center = SkyCoord(frame="galactic", l=0.0, b=0.0, unit="deg")


class KilonovaDiscReject:

    default_broker_priority = ("fink", "lasair", "alerce")

    def __init__(
        self,
        min_b: float = 5.0,
        min_bulge_sep: float = 10.0,
        max_timespan: float = 15.0,
        broker_priority: list = None,
    ):
        self.__name__ = "kilonova_disc_reject"
        self.min_b = min_b
        self.min_bulge_sep = min_bulge_sep
        self.max_timespan = max_timespan
        self.broker_priority = broker_priority or self.default_broker_priority

    def __call__(self, target: Target, t_ref: Time) -> float:
        reject = False
        exclude = False
        factors = []
        scoring_comments = []

        gal_target = target.coord.galactic

        bulge_sep = target.coord.separation(gal_center)
        if bulge_sep.deg < self.min_bulge_sep:
            reject = True
            coord_string = f"(l,b)=({gal_target.l.deg:.1f},{gal_target.b.deg:.2f})"
            scoring_comments.append(
                f"REJECT: {coord_string} too close to MW centre (<{self.min_bulge_sep:.1f}deg)"
            )

        if abs(gal_target.b.deg) < self.min_b:
            reject = True
            b_str = f"abs(b)={gal_target.b.deg:.2f}"
            scoring_comments.append(
                f"REJECT: {b_str} < {self.min_b:.2f}, too close to MW disc"
            )

        ztf_data = None
        for broker in self.broker_priority:
            data_name = f"{broker}_ztf"
            source_data = target.target_data.get(data_name, None)
            if source_data is None:
                continue
            if source_data.lightcurve is None:
                continue
            ztf_data = source_data
            break
        if ztf_data is None:
            exclude = True
            scoring_comments.append(f"no data from {self.broker_priority}")
            factors.append(1.0)
        else:
            latest_mag = ztf_data.detections["magpsf"].iloc[-1]
            latest_flux = 3631 * 10 ** (-0.4 * latest_mag)  # in Jy
            flux_factor = latest_flux * 1e6
            factors.append(flux_factor)

            flux_comment = (
                f"flux_factor={flux_factor:.2e} [uJy] from mag={latest_mag:.2f}"
            )
            scoring_comments.append(flux_comment)

            timespan = t_ref.mjd - ztf_data.detections["mjd"].min()
            if timespan > self.max_timespan:
                reject = True
                scoring_comments.append(
                    f"REJECT: too long since first detection: "
                    f"{timespan:.1f}d > {self.max_timespan}"
                )

        score = target.base_score * np.prod(factors)
        if exclude:
            score = -1.0
        if reject:
            score = -np.inf

        return score, scoring_comments
