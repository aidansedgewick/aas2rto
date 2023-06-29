import os
import time
import warnings
import yaml
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

import pandas as pd

from astropy.io import fits
from astropy.time import Time

try:
    import alerce
    from alerce.core import Alerce
except:
    alerce = None

from dk154_targets.utils import calc_file_age
from dk154_targets.target import Target, TargetData


from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets.query_managers.exc import BadKafkaConfigError, MissingObjectIdError

from dk154_targets import paths

logger = getLogger(__name__.split(".")[-1])

warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)


def process_alerce_lightcurve(
    detections: pd.DataFrame, non_detections: pd.DataFrame
) -> pd.DataFrame:
    detections["tag"] = "valid"
    dubious_mask = detections["dubious"]
    detections.loc[dubious_mask, "tag"] = "badquality"
    if not detections["candid"].is_unique:
        raise ValueError("non-unique `candid` in detections.")
    if (non_detections is not None) and (not non_detections.empty):
        non_detections["candid"] = 0
        non_detections.loc[:, "tag"] = "upperlim"
        lightcurve = pd.concat([detections, non_detections])
    else:
        lightcurve = detections
    lightcurve.sort_values("mjd", inplace=True)
    jd_dat = Time(lightcurve["mjd"].values, format="mjd").jd
    lightcurve.insert(2, "jd", jd_dat)
    return lightcurve


def target_from_alerce_lightcurve(
    objectId, lightcurve: pd.DataFrame, t_ref: Time = None
) -> Target:
    t_ref = t_ref or Time.now()
    lightcurve.sort_values("mjd", inplace=True)
    ra = lightcurve["ra"].iloc[-1]
    dec = lightcurve["dec"].iloc[-1]
    alerce_data = TargetData(lightcurve=lightcurve)
    target = Target(objectId, ra=ra, dec=dec, alerce_data=alerce_data)
    target.updated = True
    return target


class AlerceQueryManager(BaseQueryManager):
    name = "alerce"
    default_query_parameters = {
        "n_objects": 25,
        "interval": 0.25,  # how often to query for new objects
        "update": 2.0,  # how often to update each target
        "max_latest_lookback": 30.0,  # bad if latest data is older than 30 days
        "max_earliest_lookback": 70.0,  # bad if the younest data is older than 70 days (AGN!)
        "max_failed_queries": 10,  # after X failed queries, stop trying
        "max_total_query_time": 300,  # total time to spend in each stage seconds
    }
    expected_cutout_type = ("science", "template", "difference")

    def __init__(
        self,
        alerce_config: dict,
        target_lookup: Dict[str, Target],
        data_path=None,
        create_paths=True,
    ):
        self.alerce_config = alerce_config
        self.target_lookup = target_lookup

        if alerce is None:
            raise ValueError(
                "alerce module not imported correctly! "
                "either install with \033[31;1m`python3 -m pip install alerce`\033[0m, "
                "or switch `use: False` in config."
            )

        self.alerce_broker = Alerce()

        self.object_queries = self.alerce_config.get("object_queries", {})
        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.alerce_config.get("query_parameters", {})
        self.query_parameters.update(query_params)

        self.query_results = {}
        self.query_updates = {}
        self.query_results_updated = False

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def query_new_targets(
        self, t_ref: Time = None, disallow_queries=False
    ) -> pd.DataFrame:
        t_ref = t_ref or Time.now()

        for query_name, query_pattern in self.object_queries.items():
            query_results_file = self.query_results_path / f"{query_name}.csv"
            results_file_age = calc_file_age(
                query_results_file, t_ref, allow_missing=True
            )
            if results_file_age < self.query_parameters["interval"]:
                if query_results_file.stat().st_size > 1:
                    logger.debug("read")
                    query_results = pd.read_csv(query_results_file)
                    if query_results.empty:
                        continue
                    query_results.sort_values(["oid", "lastmjd"], inplace=True)
                    query_results.drop_duplicates(
                        subset="oid", keep="last", inplace=True
                    )
                    self.query_updates[query_name] = query_results.set_index(
                        "oid", verify_integrity=True
                    )
                    # It's fine that we don't make the query now (ie, don't go to "else:")
                    # we just assume that there were no results if the file is tiny.

            else:
                # if we need to make the query...
                logger.info(f"query for {query_name}")
                t_start = time.perf_counter()
                query_results = self.query_and_collate_pages(
                    query_name, query_pattern, t_ref, disallow_queries=disallow_queries
                )
                if query_results.empty:
                    query_results.to_csv(query_results_file, index=False)
                query_results.sort_values(["oid", "lastmjd"], inplace=True)
                query_results.drop_duplicates(subset="oid", keep="first", inplace=True)
                query_results.to_csv(query_results_file, index=False)
                self.query_updates[query_name] = query_results.set_index(
                    "oid", verify_integrity=True
                )
                t_end = time.perf_counter()
                logger.info(
                    f"{query_name} returned {len(query_results)} in {t_end-t_start:.1f}s"
                )
                self.query_results_updated = True

    def query_and_collate_pages(
        self, query_name, query_pattern, t_ref, disallow_queries=False
    ):
        page_results_list = []
        page_results_file_list = []
        page = 1
        n_results = self.query_parameters["n_objects"]  # init
        while n_results >= self.query_parameters["n_objects"]:
            page_results_file = self.query_results_path / f"{query_name}_{page:03d}.csv"
            if page_results_file.exists():
                # If we did this page and something broke for some reason.
                logger.debug(f"read existing {query_name} page={page}")
                page_results = pd.read_csv(page_results_file)
                page_results_list.append(page_results)
                page_results_file_list.append(page_results_file)
            else:
                if disallow_queries:
                    break
                query_data = self.prepare_query_data(query_pattern, page, t_ref)
                try:
                    logger.debug(f"query for {query_name} page={page}")
                    page_results = self.alerce_broker.query_objects(**query_data)
                    n_results = len(page_results)
                    logger.debug("")
                    page_results.to_csv(page_results_file, index=False)
                    page_results_list.append(page_results)
                    page_results_file_list.append(page_results_file)
                except Exception as e:
                    logger.info(f"query {query_name} page {page} failed")
                    page_results = []
                    print(e)
            n_results = len(page_results)
            page = page + 1

        query_results = pd.concat(page_results_list)
        for page_results_file in page_results_file_list:
            os.remove(page_results_file)  # Don't need to keep page results.
        return query_results

    def prepare_query_data(self, query_pattern: dict, page: int, t_ref: Time):
        query_data = dict(
            **query_pattern,
            page=page,
            lastmjd=t_ref.mjd - self.query_parameters["max_latest_lookback"],
            firstmjd=t_ref.mjd - self.query_parameters["max_earliest_lookback"],
            page_size=self.query_parameters["n_objects"],
        )
        return query_data

    def perform_magstats_queries(
        self, oid_list: list, t_ref: Time = None, test_input=None
    ):
        t_ref = t_ref or Time.now()
        N_existing = 0
        N_success = 0
        N_failed = 0
        t_start = time.perf_counter()
        for oid in oid_list:
            if N_failed > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed magstats queries ({N_failed}). Stop for now."
                logger.info(msg)
                break
            magstats_file = self.magstats_path / f"{oid}.csv"
            magstats_file_age = calc_file_age(magstats_file, t_ref)
            if magstats_file_age > self.query_parameters["interval"]:
                logger.debug(f"magstats query for {oid}")
                time.sleep(0.05)
                if test_input is not None:
                    assert isinstance(test_input, pd.DataFrame)
                    magstats = test_input
                    N_success = N_success + 1
                else:
                    try:
                        magstats = self.alerce_broker.query_magstats(
                            oid, format="pandas"
                        )
                        N_success = N_success + 1
                    except Exception as e:
                        N_failed = N_failed + 1
                if magstats_file.exists():
                    existing_magstats = pd.read_csv(magstats_file)
                magstats.to_csv(magstats_file, index=False)
            else:
                N_existing = N_existing + 1
        if N_success > 0:
            t_end = time.perf_counter()
            logger.info(f"magstats queries in {t_end-t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed magstats queries")
        return N_success, N_existing, N_failed

    def get_unloaded_targets_from_queries(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        existing_targets = []
        for query_name, query_updates in self.query_updates.items():
            existing_results = self.query_results.get(query_name, None)
            if existing_results is None:
                self.query_results[query_name] = query_updates
                for oid in query_updates.index:
                    existing_targets.append(oid)
        return existing_targets

    def get_updated_targets_from_queries(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        alerce_updates = []
        for query_name, updated_results in self.query_updates.items():
            existing_results = self.query_results.get(query_name, None)
            if existing_results is None:
                logger.warning(f"{query_name} results is NONE.")
                continue
            for oid, updated_row in updated_results.iterrows():
                if oid in existing_results.index:
                    existing_row = existing_results.loc[oid]
                    if updated_row["ndethist"] > existing_row["ndethist"]:
                        alerce_updates.append(oid)
                else:
                    alerce_updates.append(oid)
            self.query_results[query_name] = updated_results
        logger.info(f"{len(alerce_updates)} updated alerce targets")
        return alerce_updates

    def load_target_lightcurves(self, oid_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        loaded_lightcurves = []
        missing_lightcurves = []
        t_start = time.perf_counter()
        for oid in oid_list:
            lightcurve_file = self.get_alerce_lightcurve_file(oid)
            if not lightcurve_file.exists():
                missing_lightcurves.append(oid)
                continue
            target = self.target_lookup.get(oid, None)
            lightcurve = pd.read_csv(lightcurve_file, dtype={"candid": "Int64"})
            # TODO: not optimum to read every time... but not a bottleneck for now.
            lightcurve.sort_values("jd", inplace=True)
            if target is None:
                target = target_from_alerce_lightcurve(oid, lightcurve)
                self.target_lookup[oid] = target
                loaded_lightcurves.append(oid)
            else:
                existing_lightcurve = target.alerce_data.lightcurve
                if len(lightcurve) > len(existing_lightcurve):
                    loaded_lightcurves.append(oid)
                    target.alerce_data.add_lightcurve(lightcurve)
                    target.updated = True
                # if target.alerce_data.lightcurve.iloc[-1, "candid"]
        t_end = time.perf_counter()

        N_loaded = len(loaded_lightcurves)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} lightcurves in {t_end-t_start:.1f}s")
        return loaded_lightcurves, missing_lightcurves

    def get_lightcurves_to_update(
        self, oid_list: List[str] = None, t_ref: Time = None, magstats_requrement=True
    ):
        """
        returns a list of objectId which have alerce lightcurves are older than
        the limit.
        """
        t_ref = t_ref or Time.now()

        if oid_list is None:
            oid_list = list(self.target_lookup.keys())

        old_lightcurve_list = []
        for oid in oid_list:
            if magstats_requrement:
                magstats_file = self.magstats_path / f"{oid}.csv"
                if not magstats_file.exists():
                    continue
                magstats = pd.read_csv(magstats_file)
                if magstats.empty:
                    continue
                alerce_magmin = magstats["magmin"].min()
                if alerce_magmin is None or (not np.isfinite(alerce_magmin)):
                    continue
            lightcurve_file = self.get_alerce_lightcurve_file(oid)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if lightcurve_file_age > self.query_parameters["update"]:
                old_lightcurve_list.append(oid)
        return old_lightcurve_list

    def perform_lightcurve_queries(self, oid_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        successful_queries = []
        N_failed = 0
        logger.info(f"attempt {len(oid_list)} lightcurve queries")
        t_start = time.perf_counter()
        for oid in set(oid_list):
            lightcurve_file = self.get_alerce_lightcurve_file(oid)
            if N_failed > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({N_failed}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break
            try:
                detections = self.alerce_broker.query_detections(oid, format="pandas")
                non_detections = self.alerce_broker.query_non_detections(
                    oid, format="pandas"
                )
            except Exception as e:
                print(e)
                logger.warning(f"{oid} lightcurve query failed")
                N_failed = N_failed + 1
                continue

            lightcurve = process_alerce_lightcurve(detections, non_detections)
            if lightcurve_file.exists():
                existing_lightcurve = pd.read_csv(lightcurve_file)
            lightcurve.to_csv(lightcurve_file, index=False)
            successful_queries.append(oid)

        N_success = len(successful_queries)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed lc queries")

        return successful_queries

    def get_cutouts_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        old_cutout_list = []
        for oid, target in self.target_lookup.items():
            cutout_file = self.get_alerce_cutouts_file(oid)
            cutout_file_age = calc_file_age(cutout_file, t_ref, allow_missing=True)
            if cutout_file_age > self.query_parameters["update"]:
                old_cutout_list.append(oid)
        return old_cutout_list

    def perform_cutouts_queries(self, oid_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info(f"attempt {len(oid_list)} cutouts queries")
        successful_queries = []
        N_failed = 0
        t_start = time.perf_counter()
        for oid in set(oid_list):
            if N_failed > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({N_failed}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break
            target = self.target_lookup[oid]
            candid_col = target.alerce_data.lightcurve["candid"]
            try:
                candid = candid_col[candid_col.notna()].iloc[-1]
                if pd.isna(candid) or candid <= 0:
                    candid = None
            except Exception as e:
                candid = None
            try:
                cutouts = self.alerce_broker.get_stamps(oid, candid=candid)
            except Exception as e:
                N_failed = N_failed + 1
                continue
            cutout_file = self.get_alerce_cutouts_file(oid)
            cutouts.writeto(cutout_file, overwrite=True)
            successful_queries.append(oid)

        N_success = len(successful_queries)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"cutouts queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed cutouts queries")
        return successful_queries

    def load_cutouts(self, oid_list=None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if oid_list is None:
            oid_list = [oid for oid in self.target_lookup.keys()]

        loaded_cutouts = []
        missing_cutouts = []
        t_start = time.perf_counter()
        for oid in oid_list:
            cutout_file = self.get_alerce_cutouts_file(oid)
            if not cutout_file.exists():
                missing_cutouts.append(oid)
                continue
            target = self.target_lookup.get(oid, None)
            if target is None:
                logger.warning(f"cutouts loaded for {oid} not in target_lookup...")
                continue
            with fits.open(cutout_file) as hdul:
                for ii, hdu in enumerate(hdul):
                    cutout_type = hdu.header["STAMP_TYPE"]
                    if cutout_type not in self.expected_cutout_type:
                        logger.warning(f"{oid} stamp HDU{ii} has type '{cutout_type}'")
                    cutout_data = hdu.data
                    target.alerce_data.cutouts[cutout_type] = cutout_data
                loaded_cutouts.append(oid)
        t_end = time.perf_counter()

        N_loaded = len(loaded_cutouts)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} cutouts in {t_end-t_start:.1f}s")
        return missing_cutouts

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        self.query_new_targets(t_ref=t_ref)

        unloaded_targets = self.get_unloaded_targets_from_queries(t_ref=t_ref)
        loaded_lightcurves, missing_lightcurves = self.load_target_lightcurves(
            unloaded_targets, t_ref=t_ref
        )

        targets_with_updates = self.get_updated_targets_from_queries(t_ref=t_ref)
        targets_with_old_lightcurves = self.get_lightcurves_to_update(t_ref=t_ref)
        lightcurves_to_query = (
            missing_lightcurves + targets_with_updates + targets_with_old_lightcurves
        )
        successful_lightcurve_queries = self.perform_lightcurve_queries(
            lightcurves_to_query, t_ref=t_ref
        )
        loaded_lightcurves, missing_queried_lightcurves = self.load_target_lightcurves(
            successful_lightcurve_queries, t_ref=t_ref
        )
        if len(missing_queried_lightcurves) > 0:
            msg = f"{len(missing_queried_lightcurves)} LCs just queried are missing:"
            logger.warning(msg)
            print(missing_queried_lightcurves)

        # if self.query_results_updated:
        #     targets_with_updates = self.compare_new_query_results()
        # self.perform_magstats_queries(oid_list, t_ref=t_ref)
        # oid_lc_candidates = self.get_objects_to_query_lightcurve(oid_list)
        # self.perform_lightcurve_queries(oid_lc_candidates, t_ref=t_ref)
        # self.update_target_lightcurves(oid_lc_candidates, t_ref=t_ref)

        # Cutouts.
        missing_cutouts = self.load_cutouts()
        targets_with_old_cutouts = self.get_cutouts_to_query()
        cutouts_to_query = (
            targets_with_updates + missing_cutouts + targets_with_old_cutouts
        )
        successful_cutouts_queries = self.perform_cutouts_queries(
            cutouts_to_query, t_ref=t_ref
        )
        missing_queried_cutouts = self.load_cutouts(oid_list=successful_cutouts_queries)
        if len(missing_queried_cutouts) > 0:
            msg = f"{len(missing_queried_cutouts)} cutouts just queried are missing:"
            logger.warning(msg)
            print(missing_queried_cutouts)
