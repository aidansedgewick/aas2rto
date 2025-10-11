import collections
import pytest
import yaml
from pathlib import Path

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.exc import (
    DuplicateDataWarning,
    MissingKeysWarning,
    NotATargetError,
    UnexpectedKeysWarning,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup, group_nearby_coordinates, merge_targets


@pytest.fixture
def t_ref():
    return Time(60000.0, format="mjd")


@pytest.fixture
def empty_tl():
    return TargetLookup()


@pytest.fixture
def tl(basic_target):
    tl = TargetLookup()
    tl.add_target(basic_target)
    return tl


class Test__Init:
    def test__basic_init(self):
        # Act
        tl = TargetLookup()

        # Assert
        assert isinstance(tl.lookup, dict)
        assert isinstance(tl.id_mapping, dict)


class Test__LenMagicMethod:
    def test__len_zero(self, empty_tl: TargetLookup):
        # Assert
        assert len(empty_tl) == 0

    def test__len_one(self, tl: TargetLookup):
        # Assert
        assert len(tl) == 1  # knows there is only one unique target
        assert len(tl.id_mapping) == 2  # even though there are 2 ids for it.


class Test__ContainsMagicMethod:
    def test__existing(self, tl: TargetLookup):
        # Assert
        assert "T00" in tl

        # because...
        assert "T00" in tl.lookup

    def test__non_existing(self, tl: TargetLookup):
        # Assert
        assert "T01" not in tl

        # because...
        assert "T01" not in tl.lookup


class Test__SingleMappingUpdate:
    def test__alt_ids_are_added(self, tl: TargetLookup, basic_target: Target):
        # Arrange
        basic_target.alt_ids["src03"] = "2025aaa"

        # Act
        tl.update_id_mapping_single_target(basic_target)


class Test__SetMagicMethod:
    def test__normal_behaviour(self, basic_target: Target):
        # Arrange
        tl = TargetLookup()

        # Act
        tl["T00"] = basic_target

        # Assert
        assert "T00" in tl.lookup  # The main id
        assert isinstance(tl.lookup["T00"], Target)
        assert tl.lookup["T00"].target_id == "T00"

        assert "target_A" not in tl.lookup  # alt_id should NOT be in

        assert tl.id_mapping["T00"] == "T00"  # target_id also self-refers
        assert tl.id_mapping["target_A"] == "T00"

    def test__fails_for_non_target_obs(self, tl):
        # Act
        with pytest.raises(NotATargetError):
            tl["T01"] = "this_string_is_not_a_target"


class Test__GetMagicMethod:
    def test__get_existing_target(self, tl: TargetLookup):
        # Act
        target = tl["T00"]

        # Assert
        assert target.target_id == "T00"
        assert target.alt_ids["src02"] == "target_A"

    def test__get_existing_target_by_alt(self, tl: TargetLookup):
        # Act
        target = tl["target_A"]

        # Assert
        assert target.target_id == "T00"  # NOT the alt_id!
        assert target.alt_ids["src02"] == "target_A"


class Test__DictLikeGetMethod:
    def test__existing_target(self, tl: TargetLookup):
        # Act
        target = tl.get("T00")

        # Assert
        assert isinstance(target, Target)
        assert target.target_id == "T00"

    def test__existing_target_alt_id(self, tl: TargetLookup):
        # Act
        target = tl.get("target_A")

        # Assert
        assert isinstance(target, Target)
        assert target.target_id == "T00"

    def test__default_value(self, tl: TargetLookup):
        # Act
        target = tl.get("nonexisting")

        # Assert
        assert target is None

    def test__default_value_changed(self, tl: TargetLookup):
        # Act
        target = tl.get("nonexisting", default="a_string")

        # Assert
        assert target == "a_string"


class Test__DictLikePopMethod:
    def test__existing_target(self, tl: TargetLookup):
        # Act
        target = tl.pop("T00")

        # Assert
        assert target.target_id == "T00"
        assert "T00" not in tl
        assert "T00" not in tl.lookup
        assert "T00" not in tl.id_mapping

        # Check the alt_ids are also properly removec
        assert "target_A" in target.alt_ids.values()
        assert "target_A" not in tl.id_mapping

    def test__exiting_target_alt_id(self, tl: TargetLookup):
        # Act
        target = tl.pop("target_A")

        # Assert
        assert target.target_id == "T00"
        assert "T00" not in tl.lookup
        assert "T00" not in tl.id_mapping

        # Check the alt_ids are also properly removec
        assert "target_A" in target.alt_ids.values()
        assert "target_A" not in tl.id_mapping

    def test__default_value(self, tl: TargetLookup):
        # Act
        target = tl.pop("nonexisting")

        # Assert
        assert target is None

    def test__default_value_changed(self, tl: TargetLookup):
        # Act
        target = tl.pop("nonexisting", default="a_string")

        # Assert
        assert target == "a_string"


class Test__DictLikeKeysMethod:
    def test__type(self, tl: TargetLookup):
        # Assert
        assert isinstance(tl.keys(), collections.abc.KeysView)


class Test__DictLikeValuesMethod:
    def test__type(self, tl: TargetLookup):
        # Assert
        assert isinstance(tl.values(), collections.abc.ValuesView)


class Test__DictLikeItemsMethod:
    def test__type(self, tl: TargetLookup):
        # Assert
        assert isinstance(tl.items(), collections.abc.ItemsView)


class Test__AddTargetMethod:
    def test__normal_behaviour(self, tl: TargetLookup, other_target: Target):
        # Act
        tl.add_target(other_target)

        # Assert
        assert "T01" in tl
        assert isinstance(tl["T01"], Target)
        assert tl["T01"].target_id == "T01"
        # also check this method updates the id_mapping.
        assert "T01" in tl.id_mapping
        assert tl.id_mapping["T01"] == "T01"
        assert "target_B" in tl.id_mapping
        assert tl.id_mapping["target_B"] == "T01"

        assert len(tl) == 2

    def test__reusing_alt_id_raises_err(self, tl: TargetLookup, other_target: Target):
        # Arrange
        other_target.alt_ids["src03"] = "T00"

        # Act
        with pytest.raises(ValueError):
            tl.add_target(other_target)


class Test__UpdateAllIdMappingsMethod:
    def test__new_alt_id(self, tl: TargetLookup):
        # Arrange
        tl["T00"].alt_ids["src03"] = "2025a"

        # Act
        tl.update_target_id_mappings()

        # Assert
        assert "2025a" in tl
        assert tl.id_mapping["2025a"] == "T00"

    def test__modified_target_id(self, tl: TargetLookup):
        # Arrange
        assert "T00" in tl["T00"].alt_ids.values()  # it's also an alt_id
        tl["T00"].target_id = "T00_mod"
        assert "T00_mod" not in tl.lookup
        assert "T00_mod" not in tl.id_mapping

        # Act
        tl.update_target_id_mappings()

        # Assert
        assert len(tl.lookup) == 1  # old ref removed from "main" lookup
        assert "T00" not in tl.lookup  # old ref removed from "main" lookup
        assert "T00_mod" in tl
        assert "T00" in tl  # still an at_id

        assert "T00" in tl.id_mapping  # but still an alt_id...
        assert tl.id_mapping["T00"] == "T00_mod"  # ...which links correctly.

        assert tl.id_mapping["target_A"] == "T00_mod"
        assert tl["target_A"].target_id == "T00_mod"


class Test__UpdatePreferredIdMethod:
    def test__normal(self, tl: TargetLookup, other_target: Target):
        # Arrange
        other_target.alt_ids["best_src"] = "2025b"
        tl.add_target(other_target)
        assert tl.id_mapping["2025b"] == "T01"
        assert tl["2025b"].target_id == "T01"
        assert set(tl.lookup.keys()) == set(["T00", "T01"])

        # Act
        tl.update_to_preferred_target_id(preferred_alt="best_src")

        # Assert
        assert len(tl) == 2
        assert set(tl.keys()) == set(["T00", "2025b"])

        assert tl.id_mapping["2025b"] == "2025b"
        assert tl.id_mapping["T01"] == "2025b"


class Test__AddTargetFromFile:
    def test__normal(
        self, tl: TargetLookup, target_config_example: dict, tmp_path: Path
    ):
        # Arrange
        target_config_path = tmp_path / "test_target.yaml"
        with open(target_config_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        tl.add_target_from_file(target_config_path)

        # Assert
        assert len(tl) == 2
        assert set(tl.keys()) == set(["T00", "T99"])

        assert "T99" in tl.lookup
        assert tl["T99"].target_id == "T99"
        assert tl["target_Z"].target_id == "T99"

        assert np.isclose(tl["T99"].coord.ra.deg, 90.0)
        assert np.isclose(tl["T99"].coord.dec.deg, -30.0)

        assert np.isclose(tl["T99"].base_score, 100.0)

    def test__sexagismal_interpreted(
        self, tl: TargetLookup, target_config_example: dict, tmp_path: Path
    ):
        # Arrange
        target_config_example["ra"] = "15:00:00.00"
        target_config_example["dec"] = "-15:30:00.00"
        print(target_config_example)

        target_config_path = tmp_path / "test_target.yaml"
        with open(target_config_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        target = tl.add_target_from_file(target_config_path)

        # Assert
        assert "T99" in tl.lookup
        assert np.isclose(target.coord.ra.deg, 225.0)
        assert np.isclose(target.coord.dec.deg, -15.5)

    def test__malformed_config_returns_none(
        self, tl: TargetLookup, target_config_example: dict, tmp_path: Path
    ):
        # Arrange
        target_config_path = tmp_path / "test_target.yaml"
        target_config_example.pop("target_id")
        with open(target_config_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        with pytest.warns(MissingKeysWarning):
            target = tl.add_target_from_file(target_config_path)

        # Assert
        assert target is None

    def test__unexpected_config_returns_none(
        self, tl: TargetLookup, target_config_example: dict, tmp_path: Path
    ):
        # Arrange
        target_config_example["blah"] = 10.0
        target_config_path = tmp_path / "test_target.yaml"
        with open(target_config_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            target = tl.add_target_from_file(target_config_path)

        # Assert
        assert target is None
        assert "T99" not in tl

    def test__modify_existing(
        self, tl: TargetLookup, target_config_example: dict, tmp_path: Path
    ):
        # Arrange
        target_config_example["target_id"] = "T00"
        target_config_example["alt_ids"] = {"src03": "2025a", "src04": "cool_sn"}
        target_config_path = tmp_path / "test_target.yaml"
        with open(target_config_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        target = tl.add_target_from_file(target_config_path)

        # Assert
        assert set(tl.keys()) == set(["T00"])

        assert np.isclose(tl["T00"].coord.ra.deg, 90.0)  # coord updated!
        assert np.isclose(tl["T00"].coord.dec.deg, -30.0)

        assert np.isclose(tl["T00"].base_score, 100.0)  # base_score updated!

        assert tl["T00"].target_of_opportunity


class Test__RemoveTargetsMethod:
    def test__remove_non_finite(self, tl: TargetLookup, t_ref: Time):
        # Arrange
        tl["T00"].update_score_history(-np.inf, t_ref=t_ref)

        # Act
        removed = tl.remove_rejected_targets()

        # Assert
        assert set(tl.keys()) == set()
        assert set(tl.id_mapping.keys()) == set()  # alt_ids removed

        assert isinstance(removed, list)
        assert len(removed) == 1
        assert removed[0].target_id == "T00"

    def test__no_remove_neg_score(self, tl: TargetLookup, t_ref: Time):
        # Arrange
        tl["T00"].update_score_history(-100.0, t_ref=t_ref)

        # Act
        removed = tl.remove_rejected_targets()

        # Assert
        assert set(tl.keys()) == set(["T00"])

        assert len(removed) == 0

    def test__no_remove_non_score(self, tl: TargetLookup, t_ref: Time):
        # Act
        with pytest.warns(UserWarning):
            removed = tl.remove_rejected_targets()

        # Assert
        assert len(removed) == 0

    def test__only_subset(self, tl: TargetLookup, other_target: Target, t_ref: Time):
        # Arrange
        tl.add_target(other_target)
        tl["T00"].update_score_history(-np.inf, t_ref=t_ref)
        tl["T01"].update_score_history(-np.inf, t_ref=t_ref)

        # Act
        removed = tl.remove_rejected_targets(target_id_list=["T01"])

        # Assert
        assert len(removed) == 1
        assert set(tl.keys()) == set(["T00"])

    def test__warns_non_existing(self, tl: TargetLookup):
        # Act
        with pytest.warns(UserWarning):
            removed = tl.remove_rejected_targets(target_id_list=["Txx"])

        # Assert
        assert len(removed) == 0


class Test__GroupTargetsUtil:
    def test__bundles_grouped(self):
        # Arrange
        coord_pairs = [
            (30.0, +0.0),  # group A
            (60.0, +0.0),  # group B
            (30.0, +0.1),  # group A
            (30.0, -0.1),  # group A
            (90.0, +0.0),  # group C
            (60.1, +0.0),  # group B
        ]
        coords = SkyCoord(np.array(coord_pairs) * u.deg)

        # Act
        components = group_nearby_coordinates(coords, seplimit=1 * u.deg)

        # Assert
        assert isinstance(components, list)
        assert all([isinstance(x, list) for x in components])  # all sublists
        flat_components = [idx for grp in components for idx in grp]
        assert len(flat_components) == 6
        assert set(flat_components) == set([0, 1, 2, 3, 4, 5])

        assert len(components) == 3
        assert set([len(x) for x in components]) == set([1, 2, 3])

        expected = set([frozenset([0, 2, 3]), frozenset([1, 5]), frozenset([4])])
        result = set([frozenset(grp) for grp in components])
        assert expected == result

    def test__thin_connection(self):
        # Arrange
        coord_pairs = [(90.0, 0.0), (89.3, 0.0), (90.7, 0.0)]
        coords = SkyCoord(np.array(coord_pairs) * u.deg)
        # here 0-1 and 0-2 are connected <1deg, but 1-2 are not.
        # but they should still all group, as connected through 1.

        # Act
        components = group_nearby_coordinates(coords, seplimit=1.0 * u.deg)
        unconnected = group_nearby_coordinates(coords[1:], seplimit=1.0 * u.deg)

        # Assert
        assert len(components) == 1
        assert set(components[0]) == set([0, 1, 2])

        assert len(unconnected) == 2
        unconn_expected = set([frozenset([0]), frozenset([1])])
        unconn_result = set([frozenset(grp) for grp in unconnected])
        assert unconn_result == unconn_expected


class Test__MergeTargetsUtil:
    def test__expected_behaviour(self, basic_target: Target, other_target: Target):
        # Arrange
        basic_target.target_data["src01"] = TargetData(parameters={"z": 0.1})
        basic_target.base_score = 100.0
        other_target.target_data["src02"] = TargetData(parameters={"p": 1.0})
        other_target.target_of_opportunity = True
        other_target.alt_ids = {}

        # Act
        merged = merge_targets([basic_target, other_target])

        # Assert
        assert isinstance(merged, Target)

        assert np.isclose(merged.coord.ra.deg, 180.0)
        assert np.isclose(merged.coord.dec.deg, 0.0)

        assert np.isclose(merged.base_score, 100.0)

        assert merged.target_of_opportunity

        assert set(merged.target_data.keys()) == set(["src01", "src02"])
        assert np.isclose(merged.target_data["src01"].parameters["z"], 0.1)
        assert np.isclose(merged.target_data["src02"].parameters["p"], 1.0)

    def test__replaced_alt_ids_warns(self, basic_target: Target, other_target: Target):
        # Arrange
        basic_target.alt_ids["src03"] = "2025a"
        assert "src03" not in other_target.alt_ids  # so merged

        # Act
        with pytest.warns(DuplicateDataWarning):
            merged = merge_targets([basic_target, other_target])

        # Assert
        assert isinstance(merged, Target)

        assert merged.target_id == "T00"
        assert merged.alt_ids["src01"] == "T01"
        assert merged.alt_ids["src02"] == "target_B"
        assert merged.alt_ids["src03"] == "2025a"

    def test__replaced_target_data_warns(
        self, basic_target: Target, other_target: Target
    ):
        # Arrange
        basic_target.target_data["src01"] = TargetData(parameters={"z": 0.1})
        other_target.target_data["src01"] = TargetData(parameters={"p": 1.0})

        # Act
        with pytest.warns(DuplicateDataWarning):
            merged = merge_targets([basic_target, other_target])

        # Assert
        assert merged.target_id == "T00"
        assert set(merged.target_data["src01"].parameters.keys()) == set(["p"])

    def test__sort_by_creation_time(self, basic_target: Target, other_target: Target):
        # Arrange
        basic_target.creation_time = Time(60001.0, format="mjd")
        assert basic_target.creation_time > other_target.creation_time

        # Act
        merged = merge_targets([basic_target, other_target], sort=True)
        # 'other_target' should be 'base'

        # Assert
        assert merged.target_id == "T01"  # from 'other_target'
        assert merged.alt_ids["src01"] == "T00"  # from 'basic_target'
        assert merged.alt_ids["src02"] == "target_A"

    def test__empty_list_fails(self):
        # Act
        with pytest.raises(ValueError):
            merged = merge_targets([])

    def test__not_targets_fails(self):
        # Act
        with pytest.raises(NotATargetError):
            merged = merge_targets([None])


class Test__ConsolidateTargetsMethod:
    def test__expected_behaviour(self, tl: TargetLookup, other_target: Target):
        # Arrange
        tl["T00"].target_data["src01"] = TargetData(parameters={"z": 0.1})
        tl["T00"].target_data["src03"] = TargetData(parameters={"z": 0.2})
        coord = SkyCoord(180.0, 0.0, unit="deg")

        t00x = Target("T00x", coord=coord, source="src03")
        t00x.target_data["src03"] = TargetData(parameters={"x": 1.0})
        tl.add_target(t00x)
        tl.add_target(other_target)

        # Act
        with pytest.warns(DuplicateDataWarning):
            # expect warning when src01 is overwritte
            tl.consolidate_targets(sort=True)

        # Assert
        assert set(tl.keys()) == set(["T00", "T01"])

        assert tl.id_mapping["T00"] == "T00"
        assert tl.id_mapping["target_A"] == "T00"
        assert tl.id_mapping["T00x"] == "T00"

        assert tl["T00"].alt_ids["src01"] == "T00"
        assert tl["T00"].alt_ids["src02"] == "target_A"
        assert tl["T00"].alt_ids["src03"] == "T00x"

        assert set(tl["T00"].target_data.keys()) == set(["src01", "src03"])

    def test__no_fail_on_empty_tlookup(self, empty_tl: TargetLookup):
        # Act
        empty_tl.consolidate_targets()
