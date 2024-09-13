import warnings
from logging import getLogger
from typing import Dict, List

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

from aas2rto.exc import NotATargetError
from aas2rto.target import Target
from aas2rto.target_data import TargetData

logger = getLogger("target_lookup")


def merge_targets(*targets: Target, reorder=False):

    if reorder:
        targets = sorted(targets, key=lambda x: x.creation_time.mjd, reversed=True)

    output = targets[0]
    for target in targets[1:]:
        for key, target_data in target.target_data.items():
            output[key] = target_data
        for alt_id in target.alternative_objectIds:
            output.alternative_ids[alt_id] = output.objectId
    return output


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

    def pop(self, target_id: str, default=None):
        base_id = self.id_mapping.get(target_id, None)
        if base_id is None:
            return default
        target = self.lookup.pop(target_id)
        for alt_id in target.alternative_ids.items():
            base_id_from_alt = self.id_mapping.pop(alt_id, None)
            if base_id_from_alt is None:
                msg = f"alt_id '{alt_id}' for base_id={base_id} not in id_mapping"
                logger.warn(msg)

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

    def consolidate_targets(self, radius=5 * u.arcsec):
        raise NotImplementedError
