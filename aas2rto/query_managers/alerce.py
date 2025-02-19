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

from aas2rto import Target, TargetData
from aas2rto import utils
from aas2rto.exc import BadKafkaConfigError, MissingTargetIdError
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.utils import calc_file_age

from aas2rto import paths

logger = getLogger(__name__.split(".")[-1])

warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)

QUERY_RESULTS_COLUMNS = (
    "oid ndethist ncovhist mjdstarthist mjdendhist corrected stellar ndet".split()
    + "g_r_max g_r_max_corr g_r_mean g_r_mean_corr firstmjd lastmjd deltajd".split()
    + "meanra meandec sigmara sigmadec class classifier probability step_id_corr".split()
)


def combine_alerce_detections_non_detections(
    detections: pd.DataFrame, non_detections: pd.DataFrame, dubious_flag="dubious"
) -> pd.DataFrame:
    detections["tag"] = "valid"

    dubious_mask = detections["dubious"]
    detections.loc[dubious_mask, "tag"] = dubious_flag
    if not detections["candid"].is_unique:
        raise ValueError("non-unique `candid` in detections.")
    if (non_detections is not None) and (not non_detections.empty):
        non_detections["candid"] = -1
        if "parent_candid" in detections.columns:
            non_detections["parent_candid"] = -1
        non_detections["rfid"] = 0
        non_detections.loc[:, "tag"] = "nondet"
        lightcurve = pd.concat([detections, non_detections])
    else:
        lightcurve = detections
    lightcurve.sort_values("mjd", inplace=True)
    jd_dat = Time(lightcurve["mjd"].values, format="mjd").jd
    lightcurve.insert(2, "jd", jd_dat)
    return lightcurve


def process_alerce_lightcurve(raw_lightcurve: pd.DataFrame):
    return raw_lightcurve.copy()


def process_alerce_query_results(raw_query_updates: pd.DataFrame, comparison="lastmjd"):
    query_updates = raw_query_updates.copy()
    query_updates.sort_values(["oid", comparison], inplace=True)
    query_updates.drop_duplicates("oid", keep="last", inplace=True)
    return query_updates


def get_empty_query_results(extra_columns="lastmjd"):

    if not isinstance(extra_columns, list):
        extra_columns = list(extra_columns)
    return pd.DataFrame(columns=["oid"] + extra_columns)


def get_updates_from_query_results(
    existing_results: pd.DataFrame, updated_results: pd.DataFrame, comparison="lastmjd"
):
    """
    Compare two dataframes of query_results. Get the rows which have been updated, or are new.

    Index column should be oid
    """

    if updated_results is None or updated_results.empty:
        # Even if it's None, that's fine.
        return get_empty_query_results(extra_columns=comparison)

    updated_results = process_alerce_query_results(
        updated_results, comparison=comparison
    )

    if existing_results is None or existing_results.empty:
        return updated_results

    existing_results = process_alerce_query_results(
        existing_results, comparison=comparison
    )  # Makes a COPY.
    existing_results.set_index("oid", verify_integrity=True)

    updates = []
    for oid, updated_row in updated_results.iterrows():
        if oid not in existing_results.index:
            updates.append(updated_row)  # If we don't already know it, KEEP IT!
            continue
        existing_row = existing_results.loc[oid]
        if updated_row[comparison] > existing_row[comparison]:
            updates.append(updated_row)  # If it's been updated, KEEP IT!

    if len(updates) == 0:
        return get_empty_query_results(extra_columns=comparison)
    updates_df = pd.DataFrame(updates)
    updates_df.sort_values("oid", inplace=True)
    # updates_df.set_index("objectId", verify_integrity=True, inplace=True, drop=False)
    return updates_df


def target_from_alerce_query_row(target_id: str, data: pd.Series, t_ref: Time = None):
    if isinstance(data, dict):
        data = pd.Series(data)
    if "meanra" not in data or "meandec" not in data:
        msg = (
            f"\033[33m{target_id} target_from_alerce_query_row\033[0m"
            f"\n     missing 'meanra'/'meandec' from row {data.index}"
        )
        logger.warning(msg)
        return None
    if isinstance(data, pd.DataFrame):
        if isinstance(data["meanra"], pd.Series):
            if len(data["meanra"]) > 1:
                logger.error(
                    f"\033[31mtarget_from_alerce_query_row\033[0m has data\n{data}"
                )
                raise ValueError("your data has length greater than>1")

    ra = data["meanra"]
    dec = data["meandec"]
    alt_ids = {"ztf": target_id}
    target = Target(
        target_id, ra=ra, dec=dec, source="alerce", alt_ids=alt_ids, t_ref=t_ref
    )
    return target


def target_from_alerce_lightcurve(
    target_id, lightcurve: pd.DataFrame, t_ref: Time = None
) -> Target:
    t_ref = t_ref or Time.now()
    lightcurve.sort_values("mjd", inplace=True)
    ra = lightcurve["ra"].iloc[-1]
    dec = lightcurve["dec"].iloc[-1]
    alt_ids = {"ztf": target_id}
    target = Target(
        target_id, ra=ra, dec=dec, source="alerce", alt_ids=alt_ids, t_ref=t_ref
    )

    alerce_data = TargetData(lightcurve=lightcurve)
    target.target_data["alerce"] = alerce_data
    target.updated = True
    return target


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


def alerce_id_from_target(target: Target, resolving_order=("alerce", "ztf", "lsst")):
    for source_name in resolving_order:
        alerce_id = target.alt_ids.get(source_name, None)
        if alerce_id is not None:
            return alerce_id
    # last resort...
    target_id = target.target_id
    if target_id.lower().startswith("ztf") or target_id.lower().startswith("lsst"):
        return alerce_id  # this must be the correct id for alerce.
    return None


class AlerceQueryManager(BaseQueryManager):
    name = "alerce"
    expected_alerce_paramters = ("object_queries", "query_parameters")
    default_query_parameters = {
        "n_objects": 25,
        "object_query_interval": 0.25,  # how often to query for new objects
        "lightcurve_update_interval": 2.0,  # how often to update each target
        "object_query_firstmjd_lookback": 20.0,
        "object_query_lastmjd_lookback": None,
        "max_failed_queries": 10,  # after X failed queries, stop trying
        "max_total_query_time": 600,  # total time to spend in each stage seconds
    }
    expected_object_query_keys = (
        "classifier class_name ndet probability firstmjd lastmjd".split()
        + "ra dec radius page page_size count order_by order_mode".split()
    )
    expected_cutout_types = ("science", "template", "difference")

    def __init__(
        self,
        alerce_config: dict,
        target_lookup: Dict[str, Target],
        parent_path=None,
        create_paths=True,
    ):
        self.alerce_config = alerce_config
        self.target_lookup = target_lookup
        utils.check_unexpected_config_keys(
            self.alerce_config, self.expected_alerce_paramters, name="alerce_config"
        )

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
        utils.check_config_keys(
            self.query_parameters,
            self.default_query_parameters,
            name="alerce.query_parameters",
        )

        self.query_results = {}

        self.process_paths(parent_path=parent_path, create_paths=create_paths)

    def query_and_collate_pages(
        self, query_pattern: dict, t_ref: Time = None, query_name="object_query"
    ):
        t_ref = t_ref or Time.now()

        results_dfs = []
        page = 1
        n_results = self.query_parameters.get("n_objects", 25)
        last_n_results = n_results
        while last_n_results >= n_results:
            query_data = self.prepare_query_data(query_pattern, page, t_ref)
            name = f"alerce.query_objects.{query_name}"
            utils.check_unexpected_config_keys(
                query_data, self.expected_object_query_keys, name=name
            )
            try:
                page_results = self.alerce_broker.query_objects(**query_data)
            except Exception as e:
                msg = f"alerce query failed page={page}:\n    {type(e).__name__}: {e}"
                logger.warning(msg)
                break
            results_dfs.append(page_results)
            page = page + 1
            last_n_results = len(page_results)
        if len(results_dfs) > 0:
            return pd.concat(results_dfs)
        return pd.DataFrame(columns=["oid"])

    def prepare_query_data(self, query_pattern: dict, page: int, t_ref: Time):
        page_size = self.query_parameters["n_objects"]
        query_data = dict(**query_pattern, page=page, page_size=page_size)
        firstmjd_lookback = self.query_parameters["object_query_firstmjd_lookback"]
        if firstmjd_lookback is not None:
            query_data["firstmjd"] = t_ref.mjd - firstmjd_lookback
        lastmjd_lookback = self.query_parameters["object_query_lastmjd_lookback"]
        if lastmjd_lookback is not None:
            query_data["lastmjd"] = t_ref.mjd - lastmjd_lookback
        return query_data

    def load_existing_query_results(self, t_ref: Time = None):
        """
        returns list of query names which are outdated of date query results
        """
        t_ref = t_ref or Time.now()

        to_requery = []

        for query_name in self.object_queries.keys():
            query_results_file = self.get_query_results_file(query_name)
            query_results_file_age = calc_file_age(query_results_file, t_ref)
            if query_results_file_age > self.query_parameters["object_query_interval"]:
                # we need to do this one again.
                to_requery.append(query_name)

            if self.query_results.get(query_name) is None:  # no query data to load...
                if query_results_file.exists():
                    query_results = pd.read_csv(query_results_file)
                    query_results = process_alerce_query_results(query_results)
                    self.query_results[query_name] = query_results
        return to_requery

    def query_for_object_updates(
        self, to_requery=None, comparison="lastmjd", t_ref: Time = None
    ):
        """
        Query for updates for each fink_class in fink_config["object_queries"]
        Compare with what we currently know about - if any are new, or updated,
        return these rows.

        Parameters
        ----------
        comparison [str]:
            column_name to compare to existing results to determine updates.
            default="lastmjd"
        """
        t_ref = t_ref or Time.now()

        if to_requery is None:
            to_requery = list(self.object_queries.keys())
            logger.info(f"re-run all {len(to_requery)} queries")

        update_dfs = []
        for query_name in to_requery:
            query_pattern = self.object_queries.get(query_name, None)
            if query_pattern is None:
                msg = f"unknown query_name '{query_pattern}':\n    known {list(self.object_queries.keys())}"
                logger.warning(msg)

            query_updates = self.query_and_collate_pages(query_pattern, t_ref=t_ref)
            if query_updates is None or query_updates.empty:
                updated_results = get_empty_query_results(extra_columns=comparison)
            else:
                updated_results = process_alerce_query_results(
                    query_updates, comparison=comparison
                )
            existing_results = self.query_results.get(query_name, None)
            if existing_results is None:
                existing_results = get_empty_query_results(extra_columns=comparison)
            updates = get_updates_from_query_results(existing_results, updated_results)
            logger.info(f"{len(updates)} from '{query_name}'")

            if not updates.empty:
                update_dfs.append(updates)

            if not existing_results.empty:
                query_results = pd.concat(
                    [existing_results, updated_results], ignore_index=True
                )
            else:
                query_results = updates
            query_results = process_alerce_query_results(
                query_results, comparison=comparison
            )
            query_results_file = self.get_query_results_file(query_name)
            query_results.to_csv(
                query_results_file, index=False
            )  # Save even if empty so don't re-query
            self.query_results[query_name] = query_results

        if len(update_dfs) == 0:
            return get_empty_query_results(extra_columns=comparison)
        updated_objects = pd.concat(update_dfs, ignore_index=True)
        updated_objects = process_alerce_query_results(updated_objects)
        return updated_objects

    def new_targets_from_updates(self, updates: pd.DataFrame, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        new_targets = []
        if updates is None:
            return new_targets

        for ii, row in updates.iterrows():
            oid = row["oid"]
            target = self.target_lookup.get(oid, None)
            if target is not None:
                continue
            target = target_from_alerce_query_row(oid, row)
            if target is None:
                continue
            self.target_lookup[oid] = target
            new_targets.append(oid)
        logger.info(f"{len(new_targets)} targets added from updated queries")
        return new_targets

    def get_lightcurves_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        to_update = []
        for oid, target in self.target_lookup.items():
            lightcurve_file = self.get_lightcurve_file(oid)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if (
                lightcurve_file_age
                > self.query_parameters["lightcurve_update_interval"]
            ):
                to_update.append(oid)
        return to_update

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
                lc_data = self.alerce_broker.query_lightcurve(oid)
                det_df = pd.DataFrame(lc_data["detections"])
                non_det_df = pd.DataFrame(lc_data["non_detections"])
            except Exception as e:
                logger.error(e)
                logger.warning(f"{oid} lightcurve query failed")
                failed.append(oid)
                continue

            lightcurve = combine_alerce_detections_non_detections(det_df, non_det_df)
            if lightcurve.empty:
                logger.warning(f"\033[33m{oid} lightcurve empty!\033[0m")

            lightcurve.to_csv(lightcurve_file, index=False)
            success.append(oid)

        N_success = len(success)
        N_failed = len(failed)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed lc queries")
        return success, failed

    def load_target_lightcurves(self, oid_list: List[str] = None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        loaded = []
        missing = []
        skipped = []
        t_start = time.perf_counter()

        if oid_list is None:
            oid_list = list(self.target_lookup.keys())
            logger.info(f"try loading all {len(oid_list)} lcs in target_lookup")
        else:
            logger.info(f"try loading {len(oid_list)} lcs")

        for oid in oid_list:
            # TODO: not optimum to read every time... but not a bottleneck for now.
            lightcurve = self.load_single_lightcurve(oid)
            if lightcurve is None:
                missing.append(oid)
                logger.info(f"{oid} lightcurve is bad")
                continue

            target = self.target_lookup.get(oid, None)
            if target is None:
                logger.warning(f"load_lightcurve: {oid} not in target_lookup!")
                missing.append(oid)
                continue
            else:
                alerce_data = target.get_target_data("alerce")
                existing_lightcurve = alerce_data.lightcurve
                if existing_lightcurve is not None:
                    if len(lightcurve) <= len(existing_lightcurve):
                        skipped.append(oid)
                        continue
            loaded.append(oid)
            lightcurve = lightcurve[lightcurve["jd"] < t_ref.jd]
            alerce_data.add_lightcurve(lightcurve)
            target.updated = True
        t_end = time.perf_counter()

        N_loaded = len(loaded)
        N_missing = len(missing)
        N_skipped = len(skipped)
        # if N_loaded > 0:
        logger.info(
            f"loaded {N_loaded}, missing {N_missing} lightcurves in {t_end-t_start:.1f}s"
        )
        return loaded, missing

    def load_single_lightcurve(self, oid: str):
        t1 = time.perf_counter()

        lightcurve_file = self.get_lightcurve_file(oid)
        if not lightcurve_file.exists():
            logger.warning(f"{oid} is missing lightcurve")
            return None

        try:
            lightcurve = pd.read_csv(lightcurve_file, dtype={"candid": "Int64"})
        except pd.errors.EmptyDataError as e:
            logger.warning(f"bad lightcurve file for {oid}")
            return None
        return lightcurve

    def get_object_probabilities_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        to_query = []
        for oid in self.target_lookup.items():
            probabilities_file = self.get_probabilities_file(oid)
            prob_file_age = calc_file_age(probabilities_file, t_ref)
            if prob_file_age > self.query_parameters["object_query_interval"]:
                to_query.append(oid)
        return to_query

    def perform_object_probabilities_queries(
        self, oid_list: List[str], t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if len(oid_list) > 0:
            logger.info(f"attempt {len(oid_list)} object_prob queries")
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
                logger.warning(f"probabilities_query for {oid}: not in target_lookup!")
                failed.append(oid)
                continue
            try:
                probabilities = self.alerce_broker.query_probabilities(oid)
            except Exception as e:
                print(e)
                logger.warning(f"{oid} lightcurve query failed")
                failed.append(oid)
            prob_df = pd.DataFrame(probabilities_file)
            probabilities_file = self.get_probabilities_file(oid)
            prob_df.to_csv(probabilities_file, index=False)
        if len(success) > 0 or len(failed) > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{len(success)} successful, {len(failed)} failed prob queries")
        return success, failed

    def get_cutouts_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        to_query = []
        for oid, target in self.target_lookup.items():
            alerce_data = target.get_target_data("alerce")
            if alerce_data.detections is None:
                continue
            candid = alerce_data.detections.candid.iloc[-1]

            # If the cutouts file for our latest candid exists, we don't need to requery...
            cutout_file = self.get_cutouts_file(oid, candid, fmt="fits")
            cutout_file_age = calc_file_age(cutout_file, t_ref, allow_missing=True)
            if cutout_file_age > self.query_parameters["lightcurve_update_interval"]:
                to_query.append(oid)
        return to_query

    def perform_cutouts_queries(self, oid_list: List[str], t_ref: Time = None):
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
            alerce_data = target.get_target_data("alerce")
            if alerce_data.detections is not None:
                if "candid" in alerce_data.detections.columns:
                    candid = alerce_data.detections["candid"].iloc[-1]
                else:
                    len_det = len(alerce_data.detections)
                    msg = f"{oid} detections have no `candid` (len={len_det})"
                    logger.warning(msg)
                    continue
            try:
                cutouts = self.alerce_broker.get_stamps(oid, candid=candid)
            except Exception as e:
                failed.append(oid)
                continue
            cutout_file = self.get_cutouts_file(oid, candid, fmt="fits")
            alerce_data.meta["cutouts_candid"] = candid
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
            alerce_data = target.target_data.get("alerce", None)
            if alerce_data is None:
                continue
            if alerce_data.detections is None:
                continue

            cutouts = {}
            cutouts_candid = alerce_data.meta.get("cutouts_candid", None)
            for candid in alerce_data.detections["candid"][::-1]:
                if cutouts_candid == candid:
                    # If the existing cutouts are from this candid,
                    # they must already be the latest (as we're searching in rev.)
                    break

                cutouts_file = self.get_cutouts_file(oid, candid, fmt="fits")
                if cutouts_file.exists():
                    cutouts = cutouts_from_fits(cutouts_file)
                    break
            if len(cutouts) == 3:
                alerce_data.cutouts.update(cutouts)
                alerce_data.meta["cutouts_candid"] = candid
                loaded_cutouts.append(oid)
            if len(alerce_data.cutouts) == 0:
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

        queries_to_run = self.load_existing_query_results(t_ref=t_ref)
        updated_objects = self.query_for_object_updates(
            to_requery=queries_to_run, t_ref=t_ref
        )
        new_targets = self.new_targets_from_updates(updated_objects, t_ref=t_ref)

        # Lightcurves
        to_query = self.get_lightcurves_to_query(t_ref=t_ref)
        success, failed = self.perform_lightcurve_queries(to_query, t_ref=t_ref)
        loaded_lcs, missing_lcs = self.load_target_lightcurves(t_ref=t_ref)

        # Query probabilities of targets we've just updated.

        # Cutouts
        missing_cutouts = self.get_cutouts_to_query(t_ref=t_ref)
        success, failed = self.perform_cutouts_queries(missing_cutouts)
        self.load_cutouts()
