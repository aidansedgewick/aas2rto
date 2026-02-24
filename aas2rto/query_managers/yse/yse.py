import time
from logging import getLogger
from pathlib import Path

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto import utils
from aas2rto.query_managers.base import LightcurveQueryManager
from aas2rto.query_managers.yse.yse_client import YSEClient, YSEClientError
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


logger = getLogger(__name__.split(".")[-1])


def target_from_query_record(
    record: dict,
    target_id_col: str = "yse_id",
    coordinate_cols: tuple[str, str] = ("ra", "dec"),
    t_ref: Time = None,
):
    t_ref = t_ref or Time.now()

    target_id = record[target_id_col]
    ra_col, dec_col = coordinate_cols

    coord = SkyCoord(ra=record[ra_col], dec=record[dec_col], unit="deg")
    alt_ids = {"yse": target_id}
    return Target(target_id, coord, alt_ids=alt_ids, t_ref=t_ref)


def updates_from_explorer_queries(
    existing_results: pd.DataFrame,
    new_results: pd.DataFrame,
    id_key: str = None,
    comparison_key: str = None,
) -> pd.DataFrame:
    if id_key is None or comparison_key is None:
        raise ValueError("must provide 'id_key=<key>' and 'comparison_key=<key>'")

    sort_keys = [id_key, comparison_key]

    existing_results = existing_results.copy()
    existing_results.sort_values(sort_keys, inplace=True, ignore_index=True)
    existing_results.drop_duplicates(id_key, keep="last", inplace=True)
    existing_results.set_index(id_key, inplace=True, drop=False, verify_integrity=True)

    new_results = new_results.copy()
    new_results.sort_values(sort_keys, inplace=True, ignore_index=True)
    new_results.drop_duplicates(id_key, keep="last", inplace=True)
    new_results.set_index(id_key, inplace=True, drop=False, verify_integrity=True)

    updated_rows = []
    for target_id, new_row in new_results.iterrows():
        if target_id not in existing_results.index:
            updated_rows.append(new_row)
        else:
            existing_row = existing_results.loc[target_id]
            if new_row[comparison_key] > existing_row[comparison_key]:
                updated_rows.append(new_row)

    if len(updated_rows) == 0:
        return pd.DataFrame(columns=new_results.columns)
    return pd.DataFrame(updated_rows)


class YSEQueryManager(LightcurveQueryManager):
    name = "yse"

    id_resolving_order = ("tns", "yse")

    default_config = {
        "credentials": None,
        "explorer_queries": None,
        "explorer_query_interval": 1.0,
        "lightcurve_update_interval": 1.0,
        "max_failed_queries": 10,
        "max_query_time": 300.0,
    }

    expected_credential_keys = ("username", "password")
    expected_explorer_query_keys = (
        "query_id",
        "target_id_col",
        "coordinate_cols",
        "comparison_col",
    )

    additional_sources_lookup = {
        "ATLAS": "ATLAS",
        "P48": "P48",
        "ZTF": "P48",
        "Swift": "Swift",
        "UVOT": "Swift",
    }

    required_directories = ("lightcurves", "query_results", "parameters")

    def __init__(self, config: dict, target_lookup: TargetLookup, parent_path: Path):
        self.config = self.default_config.copy()
        self.config.update(config)

        utils.check_unexpected_config_keys(
            self.config, self.default_config, name="query_managers.yse", raise_exc=True
        )

        self.target_lookup = target_lookup

        self.process_paths(
            parent_path=parent_path, directories=self.required_directories
        )

        self.prepare_yse_query()
        self.check_query_configs()

    def prepare_yse_query(self):
        credentials = self.config.get("credentials", None)
        if credentials is None:
            raise YSEClientError(
                "query_managers.yse: must provide dict\n"
                "    credentials: {username: <>, password: <>}"  # NOT an f-string
            )
        name = "query_managers.yse.credentials"
        utils.check_unexpected_config_keys(
            credentials, self.expected_credential_keys, name=name, raise_exc=True
        )
        utils.check_missing_config_keys(
            credentials, self.expected_credential_keys, name=name, raise_exc=True
        )
        username = credentials["username"]
        password = credentials["password"]

        self.yse_client = YSEClient(username, password)

    def check_query_configs(self):
        explorer_configs: dict[str, dict] = self.config.get("explorer_queries", None)
        if explorer_configs is None:
            return

        for query_name, query_config in explorer_configs.items():
            if not isinstance(query_config, dict):
                keys_str = ",".join(f"'{x}'" for x in self.expected_explorer_query_keys)
                msg = (
                    f"config for yse query '{query_name}' should be dict with keys:\n"
                    f"    {keys_str}"
                )
                raise TypeError(msg)

            name = f"query_managers.yse.explorer_queries.{query_name}"
            exp_keys = self.expected_explorer_query_keys
            utils.check_missing_config_keys(
                query_config, exp_keys, name=name, raise_exc=True
            )
            utils.check_unexpected_config_keys(
                query_config, exp_keys, name=name, raise_exc=True
            )

    def get_query_results_filepath(self, query_name: str, fmt: str = "csv"):
        return self.paths_lookup["query_results"] / f"{query_name}.{fmt}"

    def get_lightcurve_filepath(self, yse_id: str, fmt: str = "csv"):
        return self.paths_lookup["lightcurves"] / f"{yse_id}.{fmt}"

    def explorer_queries(self, t_ref: Time = None) -> list[dict]:
        """

        Returns
        -------
        updated_target_records : list[dict]
            list of dicts - important columns from explorer_queries
            are renamed to be consistent:
                `<target_id_col>` -> `target_id`\n
                `(<coordinate_cols>)` -> `("ra", "dec")`
        """
        t_ref = t_ref or Time.now()
        interval: float = self.config["explorer_query_interval"]

        query_configs: dict = self.config.get("explorer_queries", {})

        results_list = []
        for query_name, query_config in query_configs.items():
            query_id = query_config["query_id"]
            target_id_col = query_config["target_id_col"]
            comparison_col = query_config["comparison_col"]
            ra_col, dec_col = query_config["coordinate_cols"]

            ##== Decide if we need to make the query
            query_results_filepath = self.get_query_results_filepath(query_name)
            file_age = utils.calc_file_age(query_results_filepath, t_ref)
            if file_age < interval:
                logger.info(f"{query_name} has results {file_age:.1f}d old - skip")
                continue
            logger.info(f"query for '{query_name}' updates")

            ##== Are there any existing results?
            if query_results_filepath.exists():
                existing_results = pd.read_csv(query_results_filepath)
                logger.info("will check diff against existing results")
            else:
                existing_results = pd.DataFrame(columns=[target_id_col, comparison_col])
                logger.info("no existing results")

            ##== Perform the query
            new_results = self.yse_client.query_explorer(query_id, return_type="pandas")
            if len(new_results) == 0:
                existing_results.to_csv(query_results_filepath, index=False)
                continue  # re-write file anyway so we don't re-query in 5 mins!

            ##== Combine with any existing results
            if existing_results.empty:
                combined_results = new_results
            else:
                combined_results = pd.concat([existing_results, new_results])

            ##== Only keep the lastest row for each target in the combined df
            # key_cols = [target_id_col, comparison_col]
            combined_results.sort_values([target_id_col, comparison_col], inplace=True)
            combined_results.drop_duplicates(target_id_col, keep="last", inplace=True)
            combined_results.to_csv(query_results_filepath, index=False)

            ##== Check what's now different from the existing results
            explorer_updates = updates_from_explorer_queries(
                existing_results,
                new_results,
                id_key=target_id_col,
                comparison_key=comparison_col,
            )
            explorer_updates.loc[:, "query_name"] = query_name
            if len(explorer_updates) > 0:
                column_map = {target_id_col: "yse_id", ra_col: "ra", dec_col: "dec"}
                explorer_updates.rename(column_map, axis=1, inplace=True)
                results_list.append(explorer_updates)
                logger.info(f"{len(explorer_updates)} updates for '{query_name}'")

        if len(results_list) > 0:
            results = pd.concat(results_list)
            results.drop_duplicates("yse_id", keep="first", inplace=True)
            logger.info(f"{len(results)} results from {len(results_list)} queries")
            return results.to_dict("records")
        return []

    def new_targets_from_query_records(
        self, query_records: list[dict], t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        targets_added = []
        existing_skipped = []
        for record in query_records:
            target_id = record["yse_id"]  # record keys uniform from explorer_queries()
            existing_target = self.target_lookup.get(target_id)
            if existing_target is not None:
                if target_id not in targets_added:
                    existing_skipped.append(target_id)
                continue

            target = target_from_query_record(record, t_ref=t_ref)
            self.target_lookup.add_target(target)
            targets_added.append(target_id)

        N_added = len(targets_added)
        N_skipped = len(existing_skipped)
        logger.info(f"added {N_added} new targets (skipped {N_skipped} existing)")
        return targets_added, existing_skipped

    def apply_update_messages(self, query_records: list[dict], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        t_str = t_ref.strftime("%Y-%m-%d %H:%M")

        for record in query_records:

            yse_id = record["yse_id"]  # record keys uniform from explorer_queries()
            query_name = record.get("query_name", "<unknown_query>")
            target = self.target_lookup.get(yse_id)

            msg = f"new YSE detection from '{query_name}' at {t_str}"
            target.info_messages.append(msg)

    # def select_lightcurves_to_query(self, t_ref: Time = None):
    #     t_ref = t_ref or Time.now()

    #     max_age = self.config["lightcurve_update_interval"]

    #     to_query = []
    #     no_yse_id = []
    #     for target_id, target in self.target_lookup.items():
    #         yse_id = get_yse_id_from_target(target)
    #         if yse_id is None:
    #             no_yse_id.append(target_id)
    #             continue  # There's no yse_id associated with this target.

    #         lightcurve_filepath = self.get_lightcurve_filepath(yse_id)
    #         file_age = utils.calc_file_age(lightcurve_filepath, t_ref=t_ref)
    #         if file_age > max_age:
    #             to_query.append(yse_id)

    #     logger.info(f"{len(to_query)} need querying ({len(no_yse_id)} have no YSE ID)")
    #     return to_query

    def query_lightcurves(self, id_list: list[str] = None, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        if id_list is None:
            # Can't just query all in target_lookup, some may not have valid yse_id...
            id_list = self.select_lightcurves_to_query()

        max_failed_queries = self.config["max_failed_queries"]
        max_qtime = self.config["max_query_time"]

        t_start = time.perf_counter()
        success = []
        missing = []
        failed = []
        failed_queries = 0
        for ii, yse_id in enumerate(id_list):
            # Is it sensible to conitnue with queries, or is everything failing?
            if failed_queries >= max_failed_queries:
                logger.warning(f"Too many failed LC queries ({failed_queries})")
                logger.warning(f"Stop LC queries for now")
                break
            t_elapsed = time.perf_counter() - t_start
            if t_elapsed > max_qtime:
                msg = f"LC queries taking too long ({t_elapsed:.1f}s > max {max_qtime:.1f}s)"
                logger.warning(msg)
                logger.warning("stop for now")
                break

            try:
                lightcurve = self.yse_client.query_lightcurve(yse_id)
            except Exception as e:
                msg = f"LC query {yse_id} failed:\n    {type(e).__name__}: {e}"
                logger.error(msg)
                failed.append(yse_id)
                failed_queries = failed_queries + 1
                continue

            success.append(yse_id)
            if lightcurve.empty:
                missing.append(yse_id)
            lightcurve_filepath = self.get_lightcurve_filepath(yse_id)
            lightcurve.to_csv(lightcurve_filepath)

        logger.info(f"{len(success)} LCs queried ok, {len(missing)} returned no LC")
        if len(failed) > 0:
            logger.warning(f"{len(failed)} LCs were in {failed_queries} failed queries")
        return success, failed

    def load_single_lightcurve(self, target_id: str, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        target = self.target_lookup.get(target_id, None)
        yse_id = self.get_relevant_id_from_target(target)
        if yse_id is None:
            return None

        lightcurve_filepath = self.get_lightcurve_filepath(yse_id)
        if lightcurve_filepath.exists():
            try:
                lightcurve = pd.read_csv(lightcurve_filepath)
            except Exception as e:
                msg = f"During read lc for {yse_id}:\n    {type(e).__name__}: {e}"
                logger.warning(msg)
                lightcurve = None
        else:
            lightcurve = None
        return lightcurve

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if iteration == 0:
            logger.info("no query for new targets on iter 0")

            self.load_target_lightcurves()
            return

        # Some house-keeping
        # self.update_lightcurve_filenames(self) # NotImplemented for now...

        # Are there any new/updated targets?
        updated_explorer_records = self.explorer_queries(t_ref=t_ref)
        self.new_targets_from_query_records(updated_explorer_records)
        self.apply_update_messages(updated_explorer_records)

        # Do we need to query any lightcurves?
        self.query_lightcurves()

        # Load all the lightcurves
        self.load_target_lightcurves()
