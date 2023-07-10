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

QUERY_RESULTS_COLUMNS = (
    "oid ndethist ncovhist mjdstarthist mjdendhist corrected stellar ndet".split()
    + "g_r_max g_r_max_corr g_r_mean g_r_mean_corr firstmjd lastmjd deltajd".split()
    + "meanra meandec sigmara sigmadec class classifier probability step_id_corr".split()
)


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
        if "parent_candid" in detections.columns:
            non_detections["parent_candid"] = 0
        non_detections["rfid"] = 0
        non_detections.loc[:, "tag"] = "upperlim"
        lightcurve = pd.concat([detections, non_detections])
    else:
        lightcurve = detections
    lightcurve.sort_values("mjd", inplace=True)
    jd_dat = Time(lightcurve["mjd"].values, format="mjd").jd
    lightcurve.insert(2, "jd", jd_dat)
    return lightcurve


def target_from_alerce_query_row(objectId: str, data: pd.Series):
    return Target(objectId, ra=data["meanra"], dec=data["meandec"])


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


def process_alerce_query_results(input_query_updates: pd.DataFrame):
    query_updates = input_query_updates.copy()
    query_updates.sort_values(["oid", "lastmjd"], inplace=True)
    query_updates.drop_duplicates(subset="oid", keep="last", inplace=True)
    query_updates.set_index("oid", verify_integrity=True, inplace=True)
    return query_updates


def cutouts_from_fits(cutouts_file: Path) -> Dict[str, np.ndarray]:
    cutouts_file = Path(cutouts_file)
    cutouts = {}
    with fits.open(cutouts_file) as hdul:
        for ii, hdu in enumerate(hdul):
            imtype = hdu.header["STAMP_TYPE"]
            if imtype not in AlerceQueryManager.expected_cutout_types:
                logger.warning(f"{cutouts_file.stem} stamp HDU{ii} has type '{imtype}'")
            cutouts[imtype] = hdu.data
    return cutouts


class AlerceQueryManager(BaseQueryManager):
    name = "alerce"
    default_query_parameters = {
        "n_objects": 25,
        "interval": 0.25,  # how often to query for new objects
        "update": 2.0,  # how often to update each target
        "lookback": 30.0,
        # "max_latest_lookback": 30.0,  # bad if latest data is older than 30 days
        # "max_earliest_lookback": 70.0,  # bad if the younest data is older than 70 days (AGN!)
        "max_failed_queries": 10,  # after X failed queries, stop trying
        "max_total_query_time": 600,  # total time to spend in each stage seconds
    }
    expected_cutout_types = ("science", "template", "difference")

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

        self.process_query_parameters()

        self.query_results = {}
        self.query_updates = {}
        self.query_results_updated = False

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def process_query_parameters(self):
        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.alerce_config.get("query_parameters", {})
        unknown_kwargs = [
            key for key in query_params if key not in self.query_parameters
        ]
        if len(unknown_kwargs) > 0:
            msg = f"\033[33munexpected query_parameters:\033[0m\n    {unknown_kwargs}"
            logger.warning(msg)
        self.query_parameters.update(query_params)

    def query_and_collate_pages(
        self,
        query_name,
        query_pattern,
        t_ref: Time = None,
        delete_pages=True,
        sleep=None,
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
                query_data = self.prepare_query_data(query_pattern, page, t_ref)
                try:
                    logger.debug(f"query for {query_name} page={page}")
                    page_results = self.alerce_broker.query_objects(**query_data)
                except Exception as e:
                    logger.info(f"query {query_name} page {page} failed: break")
                    page_results = pd.DataFrame([], columns=QUERY_RESULTS_COLUMNS)
                    print(e)
                    return None
                if page_results.empty:
                    break
                page_results.to_csv(page_results_file, index=False)
                page_results_list.append(page_results)
                page_results_file_list.append(page_results_file)
                if sleep is not None:
                    time.sleep(sleep)
            n_results = len(page_results)
            page = page + 1

        query_results = pd.DataFrame([], columns=QUERY_RESULTS_COLUMNS)
        if len(page_results_list) > 0:
            query_results = pd.concat(page_results_list)
        if delete_pages:
            for page_results_file in page_results_file_list:
                os.remove(page_results_file)  # Don't need to keep page results.
        return query_results

    def prepare_query_data(self, query_pattern: dict, page: int, t_ref: Time):
        query_data = dict(
            **query_pattern,
            page=page,
            firstmjd=t_ref.mjd - self.query_parameters["lookback"],
            lastmjd=t_ref.mjd,
            page_size=self.query_parameters["n_objects"],
        )
        return query_data

    def query_for_updates(self, t_ref: Time = None) -> pd.DataFrame:
        t_ref = t_ref or Time.now()

        for query_name, query_pattern in self.object_queries.items():
            query_results_file = self.get_query_results_file(query_name)
            results_file_age = calc_file_age(
                query_results_file, t_ref, allow_missing=True
            )
            if results_file_age < self.query_parameters["interval"]:
                if query_results_file.stat().st_size > 1:
                    logger.info(f"read existing {query_results_file.name}")
                    raw_query_updates = pd.read_csv(query_results_file)
                    if raw_query_updates.empty:
                        continue
                    query_updates = process_alerce_query_results(raw_query_updates)
                    self.query_updates[query_name] = query_updates
                    # It's fine that we don't make the query now (ie, don't go to "else:")
                    # we just assume that there were no results if the file is tiny.
                else:
                    continue
            else:
                # if we need to make the query...
                logger.info(f"query for {query_name}")
                t_start = time.perf_counter()
                raw_query_updates = self.query_and_collate_pages(
                    query_name, query_pattern, t_ref
                )
                if raw_query_updates is None:
                    continue
                if raw_query_updates.empty:
                    raw_query_updates.to_csv(query_results_file, index=False)
                    continue
                raw_query_updates.to_csv(query_results_file, index=False)
                query_updates = process_alerce_query_results(raw_query_updates)
                self.query_updates[query_name] = query_updates
                t_end = time.perf_counter()
                logger.info(
                    f"{query_name} returned {len(query_updates)} in {t_end-t_start:.1f}s"
                )
                self.query_results_updated = True

    def get_magstats_to_query(self, oid_list: List[str] = None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if oid_list is None:
            oid_list = list(self.target_lookup.keys())

        old_magstats = []
        for oid in oid_list:
            magstats_file = self.get_magstats_file(oid)
            magstats_file_age = calc_file_age(magstats_file, t_ref, allow_missing=True)
            if magstats_file_age > self.query_parameters["interval"]:
                old_magstats.append(oid)
        return old_magstats

    def perform_magstats_queries(self, oid_list: list, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        success = []
        failed = []

        logger.info(f"attempt {len(oid_list)} magstats queries")

        t_start = time.perf_counter()
        for oid in oid_list:
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed magstats queries ({len(failed)}). Stop for now."
                logger.info(msg)
                break
            logger.debug(f"magstats query for {oid}")
            try:
                magstats = self.alerce_broker.query_magstats(oid, format="pandas")
                success.append(oid)
            except Exception as e:
                failed.append(oid)
                print(f"failed", e)
                continue
            magstats_file = self.get_magstats_file(oid)
            magstats.to_csv(magstats_file)
        N_success = len(success)
        N_failed = len(failed)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"magstats queries in {t_end-t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed magstats queries")
        return success, failed

    def target_updates_from_query_results(self, t_ref: Time = None) -> pd.DataFrame:
        t_ref = t_ref or Time.now()

        update_dfs = []
        for query_name, updated_results in self.query_updates.items():
            # updated_results.sort_values(["oid", "lastmjd"], inplace=True)
            updated_targets = []
            existing_results = self.query_results.get(query_name, None)
            if existing_results is None:
                self.query_results[query_name] = updated_results
                logger.info(
                    f"no existing {query_name} results, use updates {len(updated_results)}"
                )
                continue
            # existing_results.sort_values(["oid", "lastmjd"], inplace=True)
            for oid, updated_row in updated_results.iterrows():
                if oid in existing_results.index:
                    existing_row = existing_results.loc[oid]
                    if updated_row["ndethist"] > existing_row["ndethist"]:
                        updated_targets.append(oid)
                else:
                    updated_targets.append(oid)
            self.query_results[query_name] = updated_results
            updated = updated_results.loc[updated_targets]
            update_dfs.append(updated)

        if len(update_dfs) > 0:
            # alerce_updates = updated_results.loc[updated_targets]
            alerce_updates = pd.concat(update_dfs)
            logger.info(f"{len(alerce_updates)} alerce targets updates")
        else:
            alerce_updates = None
        return alerce_updates

    def new_targets_from_updates(self, updates: pd.DataFrame, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        new_targets = []
        if updates is None:
            return new_targets

        for objectId, row in updates.iterrows():
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                continue
            target = target_from_alerce_query_row(objectId, row)
            self.target_lookup[objectId] = target
            new_targets.append(objectId)
        logger.info(f"{len(new_targets)} targets added from updated queries")
        return new_targets

    def get_lightcurves_to_query(
        self, oid_list: List[str] = None, t_ref: Time = None, require_magstats=True
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
            if require_magstats:
                valid_magstats = self.check_valid_magstats(oid)
                if not valid_magstats:
                    continue
            lightcurve_file = self.get_lightcurve_file(oid)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if lightcurve_file_age > self.query_parameters["update"]:
                old_lightcurve_list.append(oid)
        return old_lightcurve_list

    def check_valid_magstats(self, oid):
        magstats_file = self.get_magstats_file(oid)
        if not magstats_file.exists():
            return False
        magstats = pd.read_csv(magstats_file)
        if magstats.empty:
            return False
        alerce_magmin = magstats["magmin"].min()
        if alerce_magmin is None or (not np.isfinite(alerce_magmin)):
            return False
        return True

    def perform_lightcurve_queries(self, oid_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        success = []
        failed = []
        logger.info(f"attempt {len(oid_list)} lightcurve queries")
        t_start = time.perf_counter()
        for oid in oid_list:
            lightcurve_file = self.get_lightcurve_file(oid)
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({len(failed)}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break
            try:
                logger.debug(f"query for {oid} lc")
                detections = self.alerce_broker.query_detections(oid, format="pandas")
                non_detections = self.alerce_broker.query_non_detections(
                    oid, format="pandas"
                )
            except Exception as e:
                print(e)
                logger.warning(f"{oid} lightcurve query failed")
                failed.append(oid)
                continue

            try:
                lightcurve = process_alerce_lightcurve(detections, non_detections)
            except KeyError as e:
                logger.error(f"{oid}")
                raise ValueError
            # Why?
            # if lightcurve_file.exists():
            #     existing_lightcurve = pd.read_csv(lightcurve_file)
            lightcurve.to_csv(lightcurve_file, index=False)
            success.append(oid)

        N_success = len(success)
        N_failed = len(failed)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed lc queries")

        return success, failed

    def load_target_lightcurves(self, oid_list=None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if oid_list is None:
            oid_list = list(self.target_lookup.keys())

        loaded = []
        missing = []
        t_start = time.perf_counter()
        for oid in oid_list:
            lightcurve_file = self.get_lightcurve_file(oid)
            if not lightcurve_file.exists():
                missing.append(oid)
                continue
            target = self.target_lookup.get(oid, None)
            lightcurve = pd.read_csv(lightcurve_file, dtype={"candid": "Int64"})
            # TODO: not optimum to read every time... but not a bottleneck for now.
            if target is None:
                target = target_from_alerce_lightcurve(oid, lightcurve)
                self.target_lookup[oid] = target
            else:
                existing_lightcurve = target.alerce_data.lightcurve
                if existing_lightcurve is not None:
                    if len(lightcurve) == len(existing_lightcurve):
                        continue
            # lightcurve.sort_values("jd", inplace=True)
            loaded.append(oid)
            target.alerce_data.add_lightcurve(lightcurve)
            target.updated = True
            # if target.alerce_data.lightcurve.iloc[-1, "candid"]
        t_end = time.perf_counter()

        N_loaded = len(loaded)
        N_missing = len(missing)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} lightcurves in {t_end-t_start:.1f}s")
        if N_missing > 0:
            logger.info(f"{N_missing} lightcurves missing...")
        return loaded, missing

    def get_cutouts_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        to_query = []
        for oid, target in self.target_lookup.items():
            if target.alerce_data.detections is None:
                continue
            candid = target.alerce_data.detections.candid.iloc[-1]

            cutout_file = self.get_cutouts_file(oid, candid, fmt="fits")
            cutout_file_age = calc_file_age(cutout_file, t_ref, allow_missing=True)
            if cutout_file_age > self.query_parameters["update"]:
                to_query.append(oid)
        return to_query

    def perform_cutouts_queries(self, oid_list: List, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info(f"attempt {len(oid_list)} cutouts queries")
        success = []
        failed = []
        t_start = time.perf_counter()
        for oid in oid_list:
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({len(failed)}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break
            target = self.target_lookup.get(oid, None)
            if target is None:
                logger.warning(f"stamp_query for {oid}: not in target_lookup!")
                failed.append(oid)
                continue

            candid = None
            if target.alerce_data.detections is not None:
                if "candid" in target.alerce_data.detections.columns:
                    candid = target.alerce_data.detections["candid"].iloc[-1]
                else:
                    len_det = len(target.alerce_data.detections)
                    msg = f"{oid} detections has no `candid` (len={len_det})"
                    logger.warning(msg)
            try:
                cutouts = self.alerce_broker.get_stamps(oid, candid=candid)
            except Exception as e:
                failed.append(oid)
                continue
            cutout_file = self.get_cutouts_file(oid, candid, fmt="fits")
            target.alerce_data.meta["cutouts_candid"] = candid
            cutouts.writeto(cutout_file, overwrite=True)
            success.append(oid)

        N_success = len(success)
        N_failed = len(failed)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"cutouts queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed cutouts queries")
        return success, failed

    def load_cutouts(self, oid_list: List = None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if oid_list is None:
            oid_list = list(self.target_lookup.keys())

        loaded_cutouts = []
        missing_cutouts = []
        t_start = time.perf_counter()
        for oid, target in self.target_lookup.items():
            if target.alerce_data.detections is None:
                continue

            cutouts = {}
            cutouts_candid = target.alerce_data.meta.get("cutouts_candid", None)
            for candid in target.alerce_data.detections["candid"][::-1]:
                cutouts_file = self.get_cutouts_file(oid, candid, fmt="fits")
                if cutouts_candid == candid:
                    # If the existing cutouts are from this candid,
                    # they must already be the latest (as we're searching in rev.)
                    break
                if cutouts_file.exists():
                    cutouts = cutouts_from_fits(cutouts_file)
                    break
            if len(cutouts) == 3:
                target.alerce_data.cutouts.update(cutouts)
                target.alerce_data.meta["cutouts_candid"] = candid
                loaded_cutouts.append(oid)
            if len(target.alerce_data.cutouts) == 0:
                missing_cutouts.append(oid)
        t_end = time.perf_counter()

        N_loaded = len(loaded_cutouts)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} cutouts in {t_end-t_start:.1f}s")
        if len(missing_cutouts) > 0:
            logger.info(f"{len(missing_cutouts)} cutouts missing")
        return loaded_cutouts, missing_cutouts

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.query_for_updates(t_ref=t_ref)
        if len(self.query_results) == 0:
            # This will only happen on the initial pass.
            for query_name, query_updates in self.query_updates.items():
                new_targets = self.new_targets_from_updates(query_updates, t_ref=t_ref)
                logger.info(f"{len(new_targets)} from {query_name}")
                self.query_results[query_name] = query_updates

        magstats_to_query = self.get_magstats_to_query(t_ref=t_ref)
        self.perform_magstats_queries(magstats_to_query, t_ref=t_ref)

        # Create a new target for all the objects we've just learned about
        alerce_updates = self.target_updates_from_query_results(t_ref=t_ref)
        self.new_targets_from_updates(alerce_updates, t_ref=t_ref)

        # Do queries for updates/new targets
        updated_targets = alerce_updates.index.to_list()
        self.perform_magstats_queries(updated_targets, t_ref=t_ref)
        self.perform_lightcurve_queries(updated_targets, t_ref=t_ref)

        # Periodically lightcurves "just" in case
        to_query = self.get_lightcurves_to_query(t_ref=t_ref)
        success, failed = self.perform_lightcurve_queries(to_query, t_ref=t_ref)
        self.load_target_lightcurves()

        # Cutouts
        missing_cutouts = self.get_cutouts_to_query(t_ref=t_ref)
        success, failed = self.perform_cutouts_queries(missing_cutouts)
        self.load_cutouts()
