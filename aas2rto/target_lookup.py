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

    def __setitem__(self, key: str, target: Target) -> None:
        if not isinstance(key, str):
            logger.warning(f"type of key '{key}' should be 'str', not type={type(key)}")
        if not isinstance(target, Target):
            msg = f"new target {key} is type={type(target)}, not dk154_targets.target.Target"
            raise NotATargetError(msg)

        target_id = target.objectId
        if key != target_id:
            msg = f"__setitem__ key={key} should match target.objectId={target_id}"
            warnings.warn(UserWarning(msg))
        self.lookup[key] = target

        self.id_mapping[target_id] = target_id
        for alt_key, alt_id in target.alternative_ids.items():
            if alt_key == "target_id":
                continue
            self.id_mapping[alt_id] = target_id

    def __contains__(self, target_id: str):
        return target_id in self.id_mapping

    def add_target(self, target: Target):
        self.__setitem__(target.objectId, target)

    def get(self, target_id: str, default=None):
        """
        Behaves the same as get() method on dict.
        Return the correct target if it's a known objectId/alt_id,
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
        for source_key, alt_id in target.alternative_ids.items():
            base_id_from_alt = self.id_mapping.pop(alt_id, None)
            if base_id_from_alt is None:
                msg = f"alt_id '{alt_id}' for base_id={base_id} not in id_mapping"
                logger.warn(msg)
        return target

    def keys(self):
        return self.lookup.keys()

    def values(self):
        return self.lookup.keys()

    def items(self):
        return self.lookup.items()

    def add_target(self, target: Target):

        for alt_id in target.alternative_ids:
            if target.objectId in self.id_mapping:
                existing = self[alt_id]
                msg = (
                    f"target already exists with objectId={target.objectId}\n"
                    f"with alternative_ids={existing.alternative_ids}"
                )
                raise ValueError(msg)
        self[target.objectId] = target

    def consolidate_targets(
        self, seplimit=5 * u.arcsec, sort=False, warn_overwrite=True
    ):

        target_ids = []
        coord_list = []
        for target_id, target in self.lookup.items():
            target_ids.append(target_id)
            if target.coord is None:
                continue
            coord_list.append(target.coord)

        coords = SkyCoord(coord_list)
        grouped_indices = group_nearby_targets(coords, seplimit=seplimit)

        target_groups = [[target_ids[ii] for ii in group] for group in grouped_indices]

        N_matches = sum(len(g) > 1 for g in target_groups)
        print(target_groups)

        logger.info(f"merge {N_matches} groups of targets")

        for group in target_groups:
            if len(group) == 1:
                continue
            targets_to_merge = []
            for target_id in group:
                target_ii = self.pop(target_id)
                targets_to_merge.append(target_ii)
            merged = merge_targets(
                targets_to_merge, sort=sort, warn_overwrite=warn_overwrite
            )
            self.add_target(merged)


def group_nearby_targets(coords: SkyCoord, seplimit=5 * u.arcsec):
    """Given a SkyCoord object (array-like), return lists of target groups that
    are "connected" (within seplim).

    Not suitable for large groups >~100, as it uses recursive search.


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
    Choose the first target (ie, with coords, objectId, etc), and
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

    output = targets[0]
    for target in targets[1:]:
        print(target.target_data.keys())
        for key, target_data in target.target_data.items():
            if key in output.target_data:
                msg = (
                    f"In merge: {output.objectId}/{target.objectId}"
                    f"overwriting existing {key} target_data..."
                )
                if warn_overwrite:
                    warnings.warn(DuplicateDataWarning(msg))
            output.target_data[key] = target_data
        for alt_key, alt_id in target.alternative_ids.items():
            if alt_key == "target_id":
                key = alt_id
            else:
                key = alt_key

            existing_alt_id = output.alternative_ids.get(key, None)
            if existing_alt_id is not None and alt_id != existing_alt_id:
                msg = (
                    f"In {output.objectId}/{target.objectId} merge: "
                    f"overwrite alt_id={existing_alt_id} with key={key}"
                )
                warnings.warn(DuplicateDataWarning(msg))
            output.alternative_ids[alt_key] = alt_id
    return output
