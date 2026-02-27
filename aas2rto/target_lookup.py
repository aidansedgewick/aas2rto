import warnings
import yaml
from collections import defaultdict
from logging import getLogger
from pathlib import Path

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time

from aas2rto import utils
from aas2rto.exc import NotATargetError, DuplicateDataWarning
from aas2rto.target import Target
from aas2rto.target_data import TargetData

logger = getLogger("target_lookup")


class TargetLookup:

    EXPECTED_TARGET_CONFIG_KEYS = (
        "target_id",
        "ra",
        "dec",
        "base_score",
        "alt_ids",
        "source",
        "target_of_opportunity",
    )
    REQUIRED_TARGET_CONFIG_KEYS = ("target_id", "ra", "dec")

    def __init__(self):

        self.lookup: dict[str, Target] = {}
        self.id_mapping: dict[str, str] = {}

    def __len__(self):
        return len(self.lookup)

    def __contains__(self, target_id: str):
        # Check id_mapping NOT lookup, so any alt_id also returns frue,
        return target_id in self.id_mapping

    def update_id_mapping_single_target(self, target: Target):
        """
        All values from target.alt_ids as keys in self.id_mapping with value target.target_id
        """

        # Don't use str target_id as argument, to avoid cra
        # >>> print(tlookup.lookup)
        # {"T00": <Target t1>}
        # >>> t1.target_id = "TARGET00"
        # >>> tlookup.update_id_single_mapping(t1.target_id)
        # KeyError()

        target_id = target.target_id
        for alt_key, alt_id in target.alt_ids.items():
            self.id_mapping[alt_id] = target_id
        self.id_mapping[target_id] = target_id  # name should also refer to itself!

    def __setitem__(self, key: str, target: Target) -> None:
        if not isinstance(key, str):
            logger.warning(f"key '{key}' should be 'str', not type={type(key)}")
        if not isinstance(target, Target):
            msg = f"new target {key} is type `{type(target)}`, not `aas2rto.target.Target`"
            raise NotATargetError(msg)

        target_id = target.target_id
        if key != target_id:
            msg = f"__setitem__ key={key} should match target.target_id={target_id}"
            warnings.warn(UserWarning(msg))
        self.lookup[key] = target

        self.update_id_mapping_single_target(target)

    def __getitem__(self, key: str) -> Target:
        if not isinstance(key, str):
            logger.warning(f"key '{key}' should be 'str', not type='{type(key)}'")
        base_id = self.id_mapping.get(key, None)
        if base_id is None:
            raise KeyError(f"No target with name {key}")
        return self.lookup[base_id]

    def get(self, target_id: str, default=None):
        """
        Behaves the same as get() method on dict.
        Return the correct target if it's a known target_id/alt_id,
        else return the default value, which default=None.

        """
        if not isinstance(target_id, str):
            msg = f"key '{target_id}' should be 'str', not type='{type(target_id)}'"
            logger.warning(msg)
        base_id = self.id_mapping.get(target_id, None)
        if base_id is None:
            return default
        return self.lookup[base_id]

    def pop(self, target_id: str, default=None) -> Target:
        """
        Behaves the same as pop() method on dict.
        Return the correct target if it's a know target/alt_id,
        else return default value, which is None.
        (The target will no longer be in the target lookup!)

        All alt_ids are also removed from id_mapping.
        """

        base_id = self.id_mapping.get(target_id, None)
        if base_id is None:
            return default
        target = self.lookup.pop(base_id)

        self.id_mapping.pop(base_id)  # also remove the base id.
        removed_ids = []
        for source_key, alt_id in target.alt_ids.items():
            removed_ids.append(alt_id)
            if alt_id == base_id:
                continue  # we have already removed it.
            base_id_from_alt = self.id_mapping.pop(alt_id, None)
            if base_id_from_alt is None and alt_id not in removed_ids:
                msg = f"alt_id '{alt_id}' for base_id={base_id} not in id_mapping"
                logger.warning(msg)
        return target

    def keys(self):
        return self.lookup.keys()

    def values(self):
        return self.lookup.values()

    def items(self):
        return self.lookup.items()

    def add_target(self, target: Target):
        for src, alt_id in target.alt_ids.items():
            if alt_id in self.id_mapping:
                existing = self[alt_id]
                msg = (
                    f"target already exists with target_id={target.target_id}\n"
                    f"with alt_ids={existing.alt_ids}"
                )
                raise ValueError(msg)
        self[target.target_id] = target

    def add_target_list(self, target_list: list[Target]):
        for target in target_list:
            self.add_target(target)

    def _warn_repeat_targets_in_lookup(self):
        mem_id_values = [id(t) for t in self.lookup.values()]
        if not len(set(mem_id_values)) == len(mem_id_values):
            msg = "There are multiple references to targets in TargetLookup"
            warnings.warn(UserWarning(msg))

    def update_target_id_mappings(self):
        # Fix hanging target_ids first...
        to_modify = {}
        for target_id, target in self.lookup.items():
            if target_id != target.target_id:
                to_modify[target_id] = target.target_id

        if len(to_modify) > 0:
            logger.info("update target_id for {len(to_modify)} targets")
        for old_id, new_id in to_modify.items():
            self.lookup[new_id] = self.lookup.pop(old_id)

        # Now add any new alt_ids
        for target_id, target in self.lookup.items():
            self.update_id_mapping_single_target(target)

        self._warn_repeat_targets_in_lookup()

    def update_to_preferred_target_id(self, preferred_alt="tns"):
        to_modify = {}
        for target_id, target in self.lookup.items():
            new_id = target.alt_ids.get(preferred_alt, None)
            if new_id is None or target_id == new_id:
                continue
            target.target_id = new_id

        self.update_target_id_mappings()

    def reset_updated_targets(self, t_ref: Time = None):
        for target_id, target in self.lookup.items():
            target.updated = False
            target.send_updates = False
            target.info_messages = []

    def add_target_from_file(
        self,
        target_filepath: Path,
        t_ref: Time = None,
    ):
        """
        Load a target from yaml file, with structure:
        ```
        target_id: <str>
        ra: <float>
        dec: <float>
        base_score: <float>
        alt_ids: <dict>
        target_of_opportunity: <bool>
        """

        t_ref = t_ref or Time.now()

        target_filepath = Path(target_filepath)
        with open(target_filepath, "r") as f:
            target_config = yaml.load(f, Loader=yaml.FullLoader)
        utils.check_missing_config_keys(
            target_config,
            self.REQUIRED_TARGET_CONFIG_KEYS,
            name=target_filepath.stem,
            raise_exc=True,
        )

        utils.check_unexpected_config_keys(
            target_config,
            self.EXPECTED_TARGET_CONFIG_KEYS,
            name=target_filepath.stem,
            raise_exc=True,
        )

        target_id = target_config["target_id"]

        # Process coord properly.
        ra = target_config.pop("ra", None)  # so we can replace with 'coord'
        dec = target_config.pop("dec", None)  # so we can replace with 'coord'
        if len(str(ra).split(":")) == 3 and len(str(dec).split(":")) == 3:
            logger.info(f"interpret ({ra}, {dec}) as HH:MM:SS.ss and +DD:MM:SS.ss")
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
        else:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        target_config["coord"] = coord

        # Now do the creating/updating
        target = self.get(target_id)
        if target is not None:
            logger.info(f"{target_id} already in target list!")
            base_score = target_config.get("base_score", None)
            if base_score is not None:
                logger.info(f"updating base_score of {target_id} to {base_score}")
                target.base_score = base_score
            target.update_coordinates(coord)
            alt_ids = target_config.get("alt_ids", {})
            target.alt_ids.update(alt_ids)
            opp_target = target_config.get("target_of_opportunity", True)
            target.target_of_opportunity = opp_target
        else:
            target = Target(**target_config)
            self.add_target(target)
        return target

    def remove_rejected_targets(
        self,
        target_id_list=None,
        t_ref=None,
    ) -> list[Target]:
        t_ref = t_ref or Time.now()

        if target_id_list is None:
            target_id_list = list(self.lookup.keys())

        removed_targets = []
        for target_id in target_id_list:
            target = self.get(target_id, None)
            if target is None:
                msg = f"can't remove non-existent target '{target_id}'"
                logger.warning(msg)
                warnings.warn(msg)
                continue
            last_score = target.get_latest_science_score()  # at no_observatory.
            if last_score is None:
                msg = f"in remove_rejected: {target_id} has no score"
                logger.warning(msg)
                warnings.warn(UserWarning(msg))
                continue
            if np.isfinite(last_score):
                continue  # it can stay...
            target = self.pop(target_id)
            removed_targets.append(target)
            assert target_id not in self.id_mapping
            assert target_id not in self.lookup
        return removed_targets

    def consolidate_targets(
        self, seplimit=5 * u.arcsec, sort=False, warn_overwrite=True
    ):
        logger.info("find duplicated targets and merge...")

        target_ids = []
        coord_list = []
        for target_id, target in self.lookup.items():
            target_ids.append(target_id)
            if target.coord is None:
                continue
            coord_list.append(target.coord)

        if len(coord_list) == 0:
            logger.info("no targets in target_lookup to consolidate")
            return

        coords = SkyCoord(coord_list)
        grouped_indices = group_nearby_coordinates(coords, seplimit=seplimit)

        target_groups = [[target_ids[ii] for ii in group] for group in grouped_indices]

        N_matches = sum(len(g) > 1 for g in target_groups)
        logger.info(f"merge {N_matches} groups of targets")
        for group in target_groups:
            if len(group) == 1:
                # don't bother with merge of one target alone.
                continue
            targets_to_merge = []
            ids_to_merge = []
            for target_id in group:
                if target_id in ids_to_merge:
                    logger.warning(f"have already prepared {target_id} for merge!")
                    continue
                target_ii = self.pop(target_id)
                if target_ii is None:
                    logger.warning(f"{target_id} popped target is None!")
                    continue
                targets_to_merge.append(target_ii)
                ids_to_merge.append(target_id)
            merged = merge_targets(
                targets_to_merge, sort=sort, warn_overwrite=warn_overwrite
            )
            self.add_target(merged)

        self.update_target_id_mappings()


def group_nearby_coordinates(coords: SkyCoord, seplimit=5 * u.arcsec) -> list:
    """Given a SkyCoord object (array-like), return lists of target groups that
    are "connected" (within seplim).

    Not suitable for large groups >~100, as it uses recursive search, can possibly
    raise RecursionError

    Parameters
    ----------
    coords : `astropy.coordinates.SkyCoord`
        array-like coords of all of coords to be sorted into groups.
    seplimit : `astropy.units.Quantity`, default=<Quantity 5. arcsec>

    Returns
    -------
    components : `list[list[int]]`
        list of lists: each sublist contains the indexes into `coords` which make
        up one connected group.
        Every index is included exactly once (ie. there are some sublists which
        contain a single index, meaning that they are not grouped with any other
        targets)

    Example
    -------
    targets which are close on sky T1-T2-T3 /// T4-T5 /// T6,\n
    returns [[0,1,2], [3,4], [5]]
    """

    if not isinstance(coords, SkyCoord):
        raise TypeError(f"coords should be (array-like) SkyCoord, not {type(coords)}")

    t1, t2, sep, _ = search_around_sky(coords, coords, seplimit=seplimit)
    # second return (idx2) should be identical, last return is sep3d (irrel)

    edges = defaultdict(set)
    for ii, (e1, e2) in enumerate(zip(t1, t2)):
        edges[e1].add(e2)
        edges[e2].add(e1)

    visited = {vert: False for vert in set(edges.keys())}

    components = []
    for vert in visited.keys():
        if visited[vert] is False:
            connected = []  # connected component
            connected = dfs_util(connected, vert, edges, visited)
            components.append(connected)
    return components


def dfs_util(connected: list, vert: int, edges: dict, visited: dict, depth=0):
    """Depth First Search recursion to find
    connected components in undirected graph"""
    visited[vert] = True
    connected.append(vert)

    for next_vert in edges[vert]:
        if visited[next_vert] is False:
            connected = dfs_util(connected, next_vert, edges, visited, depth=depth + 1)
    return connected


def merge_targets(targets: list[Target], sort=False, warn_overwrite=True):
    """
    Merge all target data into a single target.
    Choose the first target (ie, with coords, target_id, etc), and
    add (overwrite) any target_data attributes sequentially
    for all the remaining targets in the list.

    Parameters
    ----------
    targets: list or tuple
        a list or tuple of any number of targets, with N>0.
        target_data from each will be combined into a single `Target`.
    sort: bool
        if true, sort list of `targets` in order target creation before merging.

    """

    if len(targets) == 0:
        raise ValueError("len of 'targets' must be >0")
    if any([not isinstance(t, Target) for t in targets]):
        raise NotATargetError(f"Some targets are not Targets!\n{targets}")

    if sort:
        targets = sorted(targets, key=lambda x: x.creation_time.mjd)

    merged_target = targets[0]
    for target in targets[1:]:
        for key, target_data in target.target_data.items():
            if key in merged_target.target_data:
                msg = (
                    f"In {merged_target.target_id}/{target.target_id} merge:"
                    f"overwriting existing {key} target_data "
                )
                if warn_overwrite:
                    warnings.warn(DuplicateDataWarning(msg))
            merged_target.target_data[key] = target_data
        if target.target_of_opportunity:
            merged_target.target_of_opportunity = True
        merged_target.base_score = max(merged_target.base_score, target.base_score)

        for alt_key, alt_id in target.alt_ids.items():
            existing_alt_id = merged_target.alt_ids.get(alt_key, None)
            if existing_alt_id is not None and alt_id != existing_alt_id:
                msg = (
                    f"In {merged_target.target_id}/{target.target_id} merge: "
                    f"overwrite alt_id={existing_alt_id} with key={alt_key}"
                )
                logger.warning(msg)
                warnings.warn(DuplicateDataWarning(msg))
            merged_target.alt_ids[alt_key] = alt_id

        merged_target.info_messages.extend(target.info_messages)
    target_ids = [t.target_id for t in targets]
    target_ids_str = ", ".join(target_ids)
    merged_target.info_messages.append(f"merged targets: {target_ids_str}")

    return merged_target
