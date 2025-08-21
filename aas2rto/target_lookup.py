import warnings
from collections import defaultdict
from logging import getLogger
from typing import Dict, List

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

from aas2rto.exc import NotATargetError, DuplicateDataWarning
from aas2rto.target import Target
from aas2rto.target_data import TargetData

logger = getLogger("target_lookup")


class TargetLookup:

    def __init__(self):

        self.lookup: Dict[str, Target] = {}
        self.id_mapping: Dict[str, str] = {}

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, key: str) -> Target:
        base_id = self.id_mapping.get(key, None)
        if base_id is None:
            raise KeyError(f"No target with name {key}")
        return self.lookup[base_id]

    def update_id_mapping_single_target(self, target: Target):

        target_id = target.target_id
        for alt_key, alt_id in target.alt_ids.items():
            self.id_mapping[alt_id] = target_id
        self.id_mapping[target_id] = target_id  # name should also refer to itself!

    def __setitem__(self, key: str, target: Target) -> None:
        if not isinstance(key, str):
            logger.warning(f"type of key '{key}' should be 'str', not type={type(key)}")
        if not isinstance(target, Target):
            msg = f"new target {key} is type={type(target)}, not dk154_targets.target.Target"
            raise NotATargetError(msg)

        target_id = target.target_id
        if key != target_id:
            msg = f"__setitem__ key={key} should match target.target_id={target_id}"
            warnings.warn(UserWarning(msg))
        self.lookup[key] = target

        self.update_id_mapping_single_target(target)

    def __contains__(self, target_id: str):
        return target_id in self.id_mapping

    def add_target(self, target: Target):
        self.__setitem__(target.target_id, target)

    def get(self, target_id: str, default=None):
        """
        Behaves the same as get() method on dict.
        Return the correct target if it's a known target_id/alt_id,
        else return the default value, which default=None.

        """
        base_id = self.id_mapping.get(target_id, None)
        if base_id is None:
            return default
        return self.lookup[base_id]

    def pop(self, target_id: str, default=None) -> Target:
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
        return self.lookup.keys()

    def items(self):
        return self.lookup.items()

    def add_target(self, target: Target):

        for alt_id in target.alt_ids:
            if target.target_id in self.id_mapping:
                existing = self[alt_id]
                msg = (
                    f"target already exists with target_id={target.target_id}\n"
                    f"with alt_ids={existing.alt_ids}"
                )
                raise ValueError(msg)
        self[target.target_id] = target

    def update_target_id_mappings(self):
        for target_id, target in self.lookup.items():
            self.update_id_mapping_single_target(target)

    def update_to_preferred_target_id(self, preferred_alt="tns"):
        to_modify = {}
        for target_id, target in self.lookup.items():
            new_id = target.alt_ids.get(preferred_alt, None)
            if new_id is None or target_id == new_id:
                continue
            to_modify[target_id] = new_id

        for old_id, new_id in to_modify.items():
            target = self.pop(old_id)
            target.target_id = new_id
            self.lookup[new_id] = target
            self.update_id_mapping_single_target(target)

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
        grouped_indices = group_nearby_targets(coords, seplimit=seplimit)

        target_groups = [[target_ids[ii] for ii in group] for group in grouped_indices]

        N_matches = sum(len(g) > 1 for g in target_groups)

        logger.info(f"merge {N_matches} groups of targets")

        for group in target_groups:
            if len(group) == 1:
                continue
            targets_to_merge = []
            for target_id in group:
                target_ii = self.pop(target_id)
                if target_ii is None:
                    logger.warning(f"{target_id} popped target is None!")
                targets_to_merge.append(target_ii)
            merged = merge_targets(
                targets_to_merge, sort=sort, warn_overwrite=warn_overwrite
            )
            self.add_target(merged)


def group_nearby_targets(coords: SkyCoord, seplimit=5 * u.arcsec):
    """Given a SkyCoord object (array-like), return lists of target groups that
    are "connected" (within seplim).

    Not suitable for large groups >~100, as it uses recursive search, can possibly
    raise RecursionError


    eg.

    for targets which are close on sky T1-T2-T3   T4-T5    T6,
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


def merge_targets(targets: List[Target], sort=False, warn_overwrite=True):
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

    if sort:
        targets = sorted(targets, key=lambda x: x.creation_time.mjd)

    if any([t is None for t in targets]):
        raise ValueError(f"Some targets are None!\n{targets}")

    output = targets[0]
    for target in targets[1:]:
        for key, target_data in target.target_data.items():
            if key in output.target_data:
                msg = (
                    f"overwriting existing {key} target_data "
                    f"in merge: {output.target_id}/{target.target_id}"
                )
                if warn_overwrite:
                    warnings.warn(DuplicateDataWarning(msg))
            output.target_data[key] = target_data
        for alt_key, alt_id in target.alt_ids.items():
            if alt_key == "target_id":
                key = alt_id
            else:
                key = alt_key

            existing_alt_id = output.alt_ids.get(key, None)
            if existing_alt_id is not None and alt_id != existing_alt_id:
                msg = (
                    f"In {output.target_id}/{target.target_id} merge: "
                    f"overwrite alt_id={existing_alt_id} with key={key}"
                )
                warnings.warn(DuplicateDataWarning(msg))
            output.alt_ids[alt_key] = alt_id

        output.update_messages.extend(target.update_messages)

    targets
    output.update_messages.append("merged targets: {}")

    return output
