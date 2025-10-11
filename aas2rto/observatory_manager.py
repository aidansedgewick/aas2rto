from logging import getLogger
from pathlib import Path
from typing import Dict

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from aas2rto import utils
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


class ObservatoryManager:

    default_config = {
        "obs_info_dt": 0.5 / 24.0,
        "obs_info_update": 2.0 / 24.0,
    }

    def __init__(
        self, config: dict, target_lookup: TargetLookup, path_manager: PathManager
    ):

        self.sites_config = config.pop("sites", {})

        self.config = self.default_config.copy()
        self.config.update(config)
        utils.check_unexpected_config_keys(
            self.config, self.default_config, "obs_manager"
        )

        self.target_lookup = target_lookup
        self.path_manager = path_manager

        self.init_observing_sites()

    def _get_empty_observing_site_lookup(self) -> Dict[str, Observer]:
        """Only for type hinting..."""
        return {}

    def init_observing_sites(self):
        self.sites = self._get_empty_observing_site_lookup()

        self.sites["no_observatory"] = None
        for site_name, site_location in self.sites_config.items():
            if isinstance(site_location, str):
                earth_location = EarthLocation.of_site(site_location)
            else:
                if "lat" not in site_location or "lon" not in site_location:
                    msg = f"{site_name}" "shoud be dict with {lat=, lon=, (height=)}"
                    logger.warning(f"Likely an exception: {msg}")
                earth_location = EarthLocation.from_geodetic(**site_location)

            self.sites[site_name] = Observer(location=earth_location, name=site_name)
        logger.info(f"init {len(self.sites)} obs sites incl. `no_observatory`")

    def compute_observatory_info(
        self, t_ref: Time = None, horizon: u.Quantity = -18 * u.deg, dt=0.5 / 24.0
    ):
        """
        Precompute some expensive information about the altitude for each observatory.

        access in a target with eg. `my_target.observatory_info["lasilla"]`
        """
        t_ref = t_ref or Time.now()
        obs_info_update = self.config["obs_info_update"]

        counter = {}
        for obs_name, observatory in self.sites.items():
            if observatory is None:
                continue
            msg = f"{obs_name}: obs_info missing/older than {obs_info_update*24:.1f}hr"
            logger.info(msg)

            counter = 0
            obs_info = ObservatoryInfo.for_observatory(
                observatory, t_ref=t_ref, horizon=horizon, dt=dt
            )
            for target_id, target in self.target_lookup.items():
                existing_info = target.observatory_info.get(obs_name)
                if existing_info is not None:
                    if t_ref.mjd - existing_info.t_ref.mjd < obs_info_update:
                        continue
                target_obs_info = obs_info.copy()
                assert target_obs_info.target_altaz is None
                if target.coord is not None:
                    target_obs_info.set_target_altaz(target.coord, observatory)
                target.observatory_info[obs_name] = target_obs_info
                if target.observatory_info[obs_name].target_altaz is None:
                    msg = f"\033[33m{target_id} {obs_name} altaz missing\033[0m"
                    logger.warning(msg)
                counter = counter + 1
            logger.info(f"updated obs_info for {counter} targets")
