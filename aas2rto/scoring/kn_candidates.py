from logging import getLogger

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto import Target

logger = getLogger("kn_score")

gal_center = SkyCoord(frame="galactic", l=0.0, b=0.0, unit="deg")


class KilonovaDiscReject:

    default_broker_priority = ("ztf", "alerce")

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
        self.broker_priority = ztf_priority or self.default_broker_priority

    def __call__(self, target: Target, observatory: Observer, t_ref: Time) -> float:
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
                f"REJECT: {coord_string} < {self.min_bulge_sep:.1f} from MW center"
            )

        if abs(gal_target.b.deg) < self.min_b:
            reject = True
            coord_ineq = f"abs(b)={gal_target.b.deg:.2f}"
            scoring_comments.append(
                f"REJECT: {coord_ineq} < {self.min_b:.2f}, too close to MW disc"
            )

        ztf_data = None
        for broker in self.broker_priority:
            data_name = f"ztf_{broker}"
            source_data = getattr(target, data_name, None)
            if source_data is None:
                continue
            if source_data.lightcurve is None:
                continue
            ztf_data = source_data
            break
        if ztf_data is None:
            exclude = True
            factors.append(1.0)
        else:
            latest_mag = ztf_data.detections["magpsf"].iloc[-1]
            latest_flux = 3631 * 10 ** (-0.4 * latest_mag)  # in Jy
            flux_factor = latest_flux * 1e9
            factors.append(flux_factor)

            flux_comment = (
                f"flux_factor={flux_factor:.2e} [nJy] from mag={latest_mag:.2f}"
            )
            scoring_comments.append(flux_comment)

            timespan = t_ref.jd - ztf_data.detections["jd"].min()
            tspan_comm = f"{timespan:.1f}d since first detection"
            if timespan > self.max_timespan:
                reject = True
                scoring_comments.append(f"REJECT: {tspan_comm} > {self.max_timespan}")

        score = target.base_score * np.prod(factors)
        if exclude:
            score = -1.0
        if reject:
            score = -np.inf

        return score, scoring_comments
