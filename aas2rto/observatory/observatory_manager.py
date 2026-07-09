import copy
import warnings
from logging import getLogger
from pathlib import Path

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from aas2rto import utils
from aas2rto.exc import MissingEphemInfoWarning
from aas2rto.observatory.ephem_info import EphemInfo
from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


class ObservatoryManager:

    default_config = {
        "dt": 15.0,
        "dt_unit": "minute",
        "horizon": -18.0,
        "sites": {},
    }

    def __init__(
        self, config: dict, target_lookup: TargetLookup, path_manager: PathManager
    ):

        self.config = self.default_config.copy()
        self.config.update(config)
        utils.check_unexpected_config_keys(
            self.config, self.default_config, "obs_manager"
        )
        self.sites_config = self.config["sites"]
        self._apply_units_to_config()

        self.target_lookup = target_lookup
        self.path_manager = path_manager

        self.current_ephem_info = {}  # The "base" ephem info, copy for each target.
        self.ephem_updated = {}

        self.init_observing_sites()

    def _apply_units_to_config(self):
        self.config["horizon"] = self.config["horizon"] * u.deg

        dt_unit = getattr(u, self.config["dt_unit"])
        if not dt_unit.is_equivalent(u.s):
            raise u.UnitTypeError(f"{dt_unit} should be equiv. to 'u.s'")
        self.config["dt"] = self.config["dt"] * dt_unit

    def init_observing_sites(self):
        self.sites: dict[str, Observer] = {}

        for site_name, site_location in self.sites_config.items():
            if isinstance(site_location, str):
                observatory = Observer.at_site(site_location)
            else:
                if "lat" not in site_location or "lon" not in site_location:
                    msg = f"{site_name}" "shoud be dict with {lat=, lon=, (height=)}"
                    logger.warning(f"Likely an exception: {msg}")
                earth_location = EarthLocation.from_geodetic(**site_location)
                observatory = Observer(location=earth_location, name=site_name)

            self.sites[site_name] = observatory
        logger.info(f"init {len(self.sites)} observatory sites")

    def update_ephem_info(self, observatory: Observer, t: Time, human_time: str = ""):
        """
        Parameters
        ----------
        observatory : Observer
        t : Time
            most usefully some local noon
        human_time
            a string only used in logging - helpful to write when ephem_info starts.
        """

        dt = self.config["dt"]
        horizon = self.config["horizon"]

        obs_name = utils.get_observatory_name(observatory)
        t_str = t.strftime("%y-%m-%d %H:%M")
        msg = f"gen. ephem_info for {obs_name}\n   starting from {t_str}"
        if human_time:
            msg = msg + f" ({human_time})"
        logger.info(msg)
        ephem_info = EphemInfo(observatory, t_ref=t, horizon=horizon, dt=dt)
        self.current_ephem_info[obs_name] = ephem_info
        self.ephem_updated[obs_name] = True
        return ephem_info

    def check_and_update_ephem_info(self, t_ref: Time = None):
        """
        Precompute some expensive information about the altitude for each observatory.
        """
        t_ref = t_ref or Time.now()

        horizon = self.config["horizon"]

        for obs_name, observatory in self.sites.items():
            if observatory is None:
                continue

            prev_noon = observatory.noon(t_ref, which="previous")
            next_noon = observatory.noon(t_ref, which="next")

            curr_ephem_info = self.current_ephem_info.get(obs_name, None)
            if curr_ephem_info is None:
                logger.info(f"no {obs_name} ephem_info")
                logger.info(f"check validity from prev. local noon first")
                curr_ephem_info = self.update_ephem_info(
                    observatory, prev_noon, human_time="prev. local noon"
                )

            # Check if any ephem data later than t_ref are still at night...
            logger.info(f"check {obs_name} ephem_info")
            future_ephem = t_ref.mjd < curr_ephem_info.t_grid.mjd  # After now
            night = curr_ephem_info.sun_altaz.alt < horizon

            remaining_night = night & future_ephem
            if sum(remaining_night) > 0:
                continue  # Still valid

            logger.info(f"{obs_name} ephem_info invalid")
            # If not valid, update the ephem_info to the next night!
            self.update_ephem_info(observatory, next_noon, human_time="next local noon")

    def apply_ephem_info(
        self,
        t_ref: Time = None,
    ):
        """
        access in a target with eg. `my_target.ephem_info["lasilla"]`
        """
        t_ref = t_ref or Time.now()

        logger.info(f"check ephem_info")
        self.check_and_update_ephem_info(t_ref=t_ref)

        for obs_name, observatory in self.sites.items():

            update_all_targets = self.ephem_updated[obs_name]
            if update_all_targets:
                logger.info(f"\n    Add new {obs_name} ephem_info to all targets")

            current_ephem_info = self.current_ephem_info.get(obs_name, None)
            if current_ephem_info is None:
                msg = f"ephem_info missing for {obs_name}..."
                logger.warning(msg)
                warnings.warn(MissingEphemInfoWarning(msg))

            counter = 0
            for target_id, target in self.target_lookup.items():

                target_ephem_info = target.ephem_info.get(obs_name, None)

                if target_ephem_info is not None and not update_all_targets:
                    continue  # No need to re-apply!
                target_ephem_info = copy.copy(current_ephem_info)  # Copy takes ~1e-5s
                target_ephem_info.set_target_altaz(target.coord)  # Takes ~0.01 sec.
                target.ephem_info[obs_name] = target_ephem_info
                counter = counter + 1
                logger.debug(f"Add {obs_name} ephem to {target_id}")

            self.ephem_updated[obs_name] = False  # Don't need to re-apply next loop...
            if counter > 0:
                logger.info(f"Update {obs_name} ephem for {counter}")
