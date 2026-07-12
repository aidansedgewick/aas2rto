import copy
from logging import getLogger
from pathlib import Path

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time, TimeDelta

from aas2rto import utils
from aas2rto.query_managers.registry import qm_registry
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.tns.tns_client import TNSClient, TNSClientError
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


class TNSCredentialError(Exception):
    pass


@qm_registry.register()  # Remember to register QM!
class TNSQueryManager(BaseQueryManager):
    name = "tns"

    default_config = {
        "credentials": None,
        "query_sleep_time": 3.0,
        "delta_lookback": 14,
        "stale_file_age": 60.0,
        "sep_limit": 5.0,
    }
    config_comments = {
        "credentials": "*REQUIRED* - a dict {'user': <tns_name>, 'uid': <tns_userid>}",
        "query_sleep_time": "",
        "sep_limit": "How close is a match? [ARCSEC]",
        "delta_lookback": "how many DAYS of deltas to search for? max available is 14",
    }

    required_credential_keys = ("user", "uid")
    required_directories = ("query_results",)

    def __init__(self, config: dict, target_lookup: TargetLookup, parent_path: Path):
        # Combine default + user config, set target_lookup + logger, process paths
        super().__init__(config, target_lookup, parent_path)

        self.process_credentials()  # sets self.tns_client
        self.apply_units_to_config()

        self.tns_results = TNSClient.get_empty_delta_results(return_type="pandas")
        self.daily_delta_results = TNSClient.get_empty_delta_results(
            return_type="pandas"
        )
        self.hourly_delta_results = TNSClient.get_empty_delta_results(
            return_type="pandas"
        )

    def apply_units_to_config(self):
        self.config["sep_limit"] = self.config["sep_limit"] * u.arcsec

    def process_credentials(self):
        credential_config = self.config.get("credentials", None)
        if credential_config is None:
            raise TNSCredentialError(
                "You must provide a dict 'credentials':"
                " {}'uid': <your user_id> and 'user': <your username>"
            )

        utils.check_missing_config_keys(
            credential_config,
            self.required_credential_keys,
            name="query_managers.tns.credentials",
            raise_exc=True,
            exc_class=TNSCredentialError,
        )

        tns_user = credential_config["user"]
        tns_uid = credential_config["uid"]

        query_sleep_time = self.config["query_sleep_time"]
        self.tns_client = TNSClient(
            tns_user, tns_uid, query_sleep_time=query_sleep_time
        )

    def get_daily_delta_filepath(self, t_ref: Time, fmt: str = "csv") -> Path:
        datestr = t_ref.strftime("%Y-%m-%d")
        return self.paths_lookup["query_results"] / f"tns_delta_{datestr}.{fmt}"

    def get_hourly_delta_filepath(self, t_ref: Time, fmt: str = "csv") -> Path:
        datestr = t_ref.strftime("%Y-%m-%d_%Hh")
        return self.paths_lookup["query_results"] / f"tns_delta_{datestr}.{fmt}"

    def get_combined_results_filepath(self, fmt: str = "csv") -> Path:
        return self.data_path / f"combined_results_latest.{fmt}"

    def load_existing_tns_results(self):
        existing_results = sorted(
            list(self.paths_lookup["query_results"].glob("tns_delta*"))
        )

        df_list = []
        for filepath in existing_results:
            self.logger.info(f"load {filepath.stem}")
            try:
                df = pd.read_csv(filepath)
            except pd.errors.EmptyDataError:
                self.logger.info(f"cannot read {filepath.stem}")
                continue
            if not df.empty:
                df_list.append(df)

        if len(df_list) > 0:
            tns_results = pd.concat(df_list, ignore_index=True)
            tns_results.sort_values(
                ["name", "lastmodified"], inplace=True, ignore_index=True
            )
            tns_results.drop_duplicates("name", keep="last", inplace=True)
        else:
            self.logger.info("no existing TNS information")
            tns_results = TNSClient.get_empty_delta_results(return_type="pandas")
        self.tns_results = tns_results

    def collect_missing_daily_deltas(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        new_deltas = []
        delta_lookback = self.config["delta_lookback"]

        for ii in range(1, int(delta_lookback) + 1):
            t_delta = t_ref - TimeDelta(24 * ii * u.hour)
            delta_filepath = self.get_daily_delta_filepath(t_delta)
            if delta_filepath.exists():
                continue
            self.logger.info(f"query for {delta_filepath.name}")
            df = self.tns_client.get_tns_daily_delta(t_delta)
            if df is not None:
                if df.empty:
                    self.logger.warning(f"{delta_filepath.name} is empty")
                df.to_csv(delta_filepath, index=False)
                new_deltas.append(df)

        if new_deltas:
            results = pd.concat(new_deltas, ignore_index=True)
        else:
            results = TNSClient.get_empty_delta_results(return_type="pandas")
        self.daily_delta_results = results
        return results

    def collect_missing_hourly_deltas(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        glob_patten = "tns_delta*h.csv"  # search for any date with 'h' in.
        existing_hourly_deltas = sorted(
            list(self.paths_lookup["query_results"].glob(glob_patten))
        )

        # Remove any hourly deltas from yday, day before, etc.
        relevant_stem = self.get_daily_delta_filepath(t_ref).stem
        for filepath in existing_hourly_deltas:
            if relevant_stem in filepath.name:
                continue  # it's still relevant
            self.logger.info(f"unlink {filepath.name}")
            filepath.unlink()  # Otherwise it's old, and we'll have the daily one.

        remaining_hourly_deltas = sorted(
            list(self.paths_lookup["query_results"].glob(glob_patten))
        )
        self.logger.info(
            f"there are {len(remaining_hourly_deltas)} hourly deltas already"
        )

        ref_dt = t_ref.datetime
        curr_hour = ref_dt.hour
        t_floor_data = dict(year=ref_dt.year, month=ref_dt.month, day=ref_dt.day)
        t_floor = Time(t_floor_data)
        new_deltas = []
        for hour in range(curr_hour + 1):
            t_delta = t_floor + hour * u.hour
            delta_filepath = self.get_hourly_delta_filepath(t_delta)
            if delta_filepath.exists():
                continue
            self.logger.info(f"query for {delta_filepath}")
            df = self.tns_client.get_tns_hourly_delta(hour)
            if df is None:
                continue
            df.to_csv(delta_filepath, index=False)
            new_deltas.append(df)

        if new_deltas:
            results = pd.concat(new_deltas)
        else:
            results = TNSClient.get_empty_delta_results(return_type="pandas")
        self.hourly_delta_results = results
        return results

    def collect_missing_tns_delta_files(self, t_ref: Time = None):
        self.collect_missing_daily_deltas(t_ref=t_ref)
        self.collect_missing_hourly_deltas(t_ref=t_ref)

    def combine_delta_results(self) -> None:
        """
        Concatenate the existing results, any daily_delta which has just been
        dowloaded, and any hourly_delta which has just been downloaded.

        Sort by name and lastmodified, and keep only the last modified.
        """
        to_combine = [
            self.tns_results,
            self.daily_delta_results,
            self.hourly_delta_results,
        ]
        to_combine = [df for df in to_combine if not df.empty]
        if len(to_combine) == 0:
            # Nothing to combine!
            self.logger.info("no results to combine!")
            self.tns_results = TNSClient.get_empty_delta_results(return_type="pandas")
            return

        combined = pd.concat(to_combine, ignore_index=True)
        combined.sort_values(["name", "lastmodified"], inplace=True)
        combined.drop_duplicates("name", keep="last")
        combined_results_filepath = self.get_combined_results_filepath()
        combined.to_csv(combined_results_filepath)
        self.tns_results = combined

    def match_on_coordinates(
        self,
        results: pd.DataFrame = None,
        seplimit: u.Quantity = None,
        ra_col: str = "ra",
        dec_col: str = "declination",
        t_ref: Time = None,
    ) -> None:

        t_ref = t_ref or Time.now()
        results = results or self.tns_results

        if seplimit is None:
            # Can't use one-liner here - u.Quantity does not like bool comparison??
            seplimit = self.config["sep_limit"]

        if results is None or results.empty:
            self.logger.info("no results to match!")
            return

        tns_candidate_coords = SkyCoord(
            ra=results[ra_col], dec=results[dec_col], unit=u.deg
        )

        target_candidate_coords = []
        target_candidate_target_ids = []
        for target_id, target in self.target_lookup.items():
            target_candidate_coords.append(target.coord)
            target_candidate_target_ids.append(target_id)
        if len(target_candidate_coords) == 0:
            self.logger.info("no targets in target_lookup to match!")
            return
        msg = (
            f"Match {len(results)} TNS rows "
            f"against {len(target_candidate_coords)} targets "
            f"(limit <{seplimit})"
        )
        self.logger.info(msg)

        target_candidate_coords = SkyCoord(target_candidate_coords)

        target_match_idx, tns_match_idx, skysep, _ = search_around_sky(
            target_candidate_coords, tns_candidate_coords, seplimit
        )
        self.logger.info(f"coordinate match for {len(target_match_idx)} TNS objects")

        for ii, (idx1, idx2, skysep) in enumerate(
            zip(target_match_idx, tns_match_idx, skysep)
        ):
            target_id = target_candidate_target_ids[idx1]
            tns_parameters = self.tns_results.iloc[idx2].to_dict()
            self.update_tns_target_data(target_id, tns_parameters, skysep=skysep)
        return

    def update_tns_target_data(
        self, target_id: str, tns_parameters: dict, skysep: u.Quantity = None
    ):

        target = self.target_lookup[target_id]

        tns_data = target.get_target_data("tns")
        curr_tns_name = target.alt_ids.get("tns", None)

        tns_name = tns_parameters["name"]
        if curr_tns_name is not None and tns_name != curr_tns_name:
            msg = (
                f"{target_id} new tns match {tns_name} "
                f"does not match old tns_match {curr_tns_name}"
            )
            self.logger.warning(msg)

        # Who refers to targets by TNS name only?
        tns_alt_keys = ["tns", "yse"]

        # Who has their own internal names for targets?
        alt_id_prefixes = {
            "ZTF": "ztf",
            "ATLAS": "atlas",
            "PS": "panstarrs",
            "LSST": "iau_lsst",
        }

        for alt_key in tns_alt_keys:
            target.alt_ids[alt_key] = tns_name

        internal_names = (
            str(tns_parameters["internal_names"]).replace(" ", "").split(",")
        )
        for name in internal_names:
            for prefix, alt_key in alt_id_prefixes.items():
                if name.startswith(prefix):
                    curr_alt_id = target.alt_ids.get(alt_key)
                    if curr_alt_id is not None and curr_alt_id != name:
                        msg = (
                            f"{target_id}/{tns_name}: new {alt_key} id "
                            f"'{name}' does not match existing '{curr_alt_id}'"
                        )
                        self.logger.warning(msg)
                    target.alt_ids[alt_key] = name

        existing_parameters = tns_data.parameters or {}
        keep_keys = (
            "objid redshift name name_prefix type "
            "discoverydate lastmodified ".split()
        )
        updated_parameters = {k: tns_parameters[k] for k in keep_keys}
        updated_parameters["skysep"] = skysep
        tns_data.parameters = updated_parameters

        prev_modified = existing_parameters.get("lastmodified", "")  # an ISO str
        if updated_parameters["lastmodified"] > prev_modified:
            target.updated = True

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if iteration == 0:
            # Just load and match in startup - no new queries yet...
            self.load_existing_tns_results()
            self.combine_delta_results()  # Should really do nothing except write here.
            self.match_on_coordinates()
            return

        self.collect_missing_tns_delta_files(t_ref=t_ref)
        self.combine_delta_results()
        self.match_on_coordinates()
        self.clear_stale_files(stale_age=self.config["stale_file_age"])
