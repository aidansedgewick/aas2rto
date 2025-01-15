import pytest

import numpy as np

from astropy import units as u
from astropy.time import Time

from aas2rto.exc import NotATargetError
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup, merge_targets


@pytest.fixture
def mock_target():
    return Target("ZTF00abc", 30.0, 45.0)


@pytest.fixture
def mock_target_alt_ids():
    ztf_name = "ZTF01def"
    alt_names = {"ztf": ztf_name, "tns": "SN2001A"}
    return Target(ztf_name, 60.0, -30.0, alt_ids=alt_names)


@pytest.fixture
def target_lookup_one_target(mock_target):
    tl = TargetLookup()
    tl[mock_target.objectId] = mock_target
    return tl


@pytest.fixture
def target_lookup_two_targets(mock_target, mock_target_alt_ids):
    t1 = mock_target
    t2 = mock_target_alt_ids

    tl = TargetLookup()
    tl[t1.objectId] = t1
    tl[t2.objectId] = t2
    return tl


class Test__MergeTargets:
    def test__two_targets_sort_true(self):
        t_ref01 = Time(60000.0, format="mjd")
        t_ref02 = Time(60010.0, format="mjd")
        t1_alt_ids = {"alt01": "SN00"}
        t2_alt_ids = {"alt02": "TR_AAA"}

        t1 = Target(
            "T001", ra=45.0, dec=30.0, source="src01", alt_ids=t1_alt_ids, t_ref=t_ref01
        )
        t2 = Target(
            "T002", ra=30.0, dec=16.0, source="src02", alt_ids=t2_alt_ids, t_ref=t_ref02
        )

        output = merge_targets([t2, t1], sort=True)

        assert np.isclose(output.ra, 45.0)  # ie, keep coords from the first one.
        assert np.isclose(output.dec, 30.0)

        assert output.objectId == "T001"

        assert output.alt_ids["src01"] == "T001"
        assert output.alt_ids["src02"] == "T002"


class Test__TargetLookupInit:

    def test__normal_behaviour(self):
        tlookup = TargetLookup()

        assert isinstance(tlookup.lookup, dict)
        assert isinstance(tlookup.id_mapping, dict)


class Test__DunderGetSet:

    def test__dunder_setitem_normal(self, mock_target):
        tlookup = TargetLookup()
        tlookup["ZTF00abc"] = mock_target

        assert "ZTF00abc" in tlookup.lookup.keys()
        assert isinstance(tlookup.lookup["ZTF00abc"], Target)
        assert set(tlookup.id_mapping.keys()) == set(["ZTF00abc"])
        assert tlookup.id_mapping["ZTF00abc"] == "ZTF00abc"
        assert len(tlookup.id_mapping) == 1

    def test__dunder_setitem_alt_ids(self, mock_target_alt_ids):
        tlookup = TargetLookup()

        tlookup["ZTF01def"] = mock_target_alt_ids

        assert set(tlookup.lookup.keys()) == set(["ZTF01def"])
        assert isinstance(tlookup.lookup["ZTF01def"], Target)
        assert set(tlookup.id_mapping.keys()) == set(["ZTF01def", "SN2001A"])

        assert tlookup.id_mapping["ZTF01def"] == "ZTF01def"
        assert tlookup.id_mapping["SN2001A"] == "ZTF01def"

    def test__dunder_setitem_two_targets(self, mock_target, mock_target_alt_ids):
        tlookup = TargetLookup()
        t1 = mock_target
        t2 = mock_target_alt_ids

        tlookup[t1.objectId] = mock_target
        tlookup[t2.objectId] = mock_target_alt_ids

        assert set(tlookup.lookup.keys()) == set(["ZTF00abc", "ZTF01def"])
        assert isinstance(tlookup.lookup["ZTF00abc"], Target)
        assert isinstance(tlookup.lookup["ZTF01def"], Target)
        assert tlookup.lookup["ZTF00abc"].objectId == "ZTF00abc"
        assert tlookup.lookup["ZTF01def"].objectId == "ZTF01def"

        assert set(tlookup.id_mapping.keys()) == set(
            ["ZTF00abc", "ZTF01def", "SN2001A"]
        )
        assert tlookup.id_mapping["ZTF00abc"] == "ZTF00abc"
        assert tlookup.id_mapping["ZTF01def"] == "ZTF01def"
        assert tlookup.id_mapping["SN2001A"] == "ZTF01def"

    def test__dunder_set_fails_on_not_target_cls(self):
        tlookup = TargetLookup()

        with pytest.raises(NotATargetError):
            tlookup["should_fail"] = 0

    def test__dunder_get_normal(self, target_lookup_two_targets):
        tlookup = target_lookup_two_targets

        t1 = tlookup["ZTF00abc"]
        assert isinstance(t1, Target)
        assert t1.objectId == "ZTF00abc"

        t2a = tlookup["ZTF01def"]
        assert t2a.objectId == "ZTF01def"
        t2b = tlookup["SN2001A"]
        assert t2b.objectId == "ZTF01def"
        assert t2a is t2b


class Test__OtherDunders:
    def test__dunder_len(self, target_lookup_one_target, target_lookup_two_targets):
        tlookup1 = TargetLookup()
        assert len(tlookup1) == 0

        assert len(target_lookup_one_target) == 1
        assert len(target_lookup_one_target.id_mapping) == 1

        assert len(target_lookup_two_targets) == 2
        assert len(target_lookup_two_targets.id_mapping) == 3

    def test__dunder_contains(self, target_lookup_two_targets):
        tlookup = target_lookup_two_targets

        assert "ZTF00abc" in tlookup
        assert "ZTF01def" in tlookup

        assert "SN2001A" not in tlookup.lookup
        assert "SN2001A" in tlookup  # Because it's an alt_id of ZTF01def

        assert "made_up_targ" not in tlookup


class Test__GetMethod:
    def test__get_method(self, target_lookup_one_target: TargetLookup):
        tlookup = target_lookup_one_target

        t1 = tlookup.get("ZTF00abc")
        assert isinstance(t1, Target)

    def test__get_method_with_alt_name(self, target_lookup_two_targets: TargetLookup):
        tlookup = target_lookup_two_targets

        t1a = tlookup.get("ZTF01def")
        assert isinstance(t1a, Target)
        assert t1a.objectId == "ZTF01def"
        t1b = tlookup.get("SN2001A")
        assert isinstance(t1b, Target)
        assert t1b.objectId == "ZTF01def"  # it should be the same target...
        assert t1a is t1b  # it IS the same target!

    def test__get_method_missing_target(self):
        tlookup = TargetLookup()

        t1 = tlookup.get("some_target", -1)
        assert t1 == -1


class Test__PopMethod:
    def test__pop_method(self, target_lookup_two_targets: TargetLookup):
        tlookup = target_lookup_two_targets
        assert len(tlookup) == 2

        assert "ZTF00abc" in tlookup
        assert "ZTF01def" in tlookup
        assert set(tlookup.lookup.keys()) == set(["ZTF00abc", "ZTF01def"])
        assert "SN2001A" in tlookup

        removed = tlookup.pop("ZTF01def")
        assert removed.objectId == "ZTF01def"

        assert len(tlookup) == 1
        assert "ZTF01def" not in tlookup.lookup
        assert "ZTF01def" not in tlookup
        assert "SN2001A" not in tlookup

    def test__pop_with_alt_id(sef, target_lookup_two_targets: TargetLookup):
        tlookup = target_lookup_two_targets

        removed = tlookup.pop("SN2001A")
        assert len(tlookup) == 1
        assert "ZTF01def" not in tlookup
        assert "SN2001A" not in tlookup

        assert removed.objectId == "ZTF01def"


class Test__AddTarget:

    def test__add_target(self):

        alt_ids = {"datasrc": "02xyz", "othersrc": "AT_001"}
        new_targ = Target("ZTF02ijk", 45.0, -45.0, alt_ids=alt_ids)

        tlookup = TargetLookup()

        tlookup.add_target(new_targ)

        assert len(tlookup) == 1
        assert "ZTF02ijk" in tlookup
        assert "02xyz" in tlookup
        assert "AT_001" in tlookup

        assert "datasrc" not in tlookup
        assert "othersrc" not in tlookup


class Test__UpdateIdMappings:
    def test__update_target_id_mappings(self, mock_target_alt_ids):

        tl = TargetLookup()

        # this is a bad way to update targets!
        tl.lookup["ZTF01def"] = mock_target_alt_ids
        assert "ZTF01def" not in tl.id_mapping
        assert "SN2001A" not in tl.id_mapping

        tl.update_id_mapping_single_target(mock_target_alt_ids)
        assert set(tl.id_mapping.keys()) == set(["ZTF01def", "SN2001A"])
        assert tl.id_mapping["ZTF01def"] == "ZTF01def"
        assert tl.id_mapping["SN2001A"] == "ZTF01def"

    def update_all_target_id_mappings(self, mock_target_alt_ids):
        tl = TargetLookup()
        tl.add_target(mock_target_alt_ids)

        assert set(tl.id_mapping.keys()) == set(["ZTF01def", "SN2001A"])
        assert tl.id_mapping["ZTF01def"] == "ZTF01def"
        assert tl.id_mapping["SN2001A"] == "ZTF01def"

        mock_target_alt_ids.alt_ids["survey_X"] = "X_001"

        assert tl["ZTF01def"].alt_ids["survey_X"] == "X_001"  # it's there...
        assert "X_001" not in tl.id_mapping  # but tl doesn't know about it yet!
        assert "X_001" not in tl  # use __contains__

        tl.update_target_id_mappings()

        # Now TL should know about it!
        assert "X_001" in tl.id_mapping
        assert "X_001" in tl
        assert tl["X_001"].objectId == "ZTF01def"


class Test__ConsolidateTargets:

    def test__simple(self):

        tl = TargetLookup()
        t1 = Target("T101", 45.0, 0.0, alt_ids={"src01": "T101"})
        t2 = Target("AAAA", 45.0, 0.0, alt_ids={"src02": "AAAA"})

        tl.add_target(t1)
        tl.add_target(t2)

        assert tl.id_mapping["T101"] == "T101"
        assert tl.id_mapping["AAAA"] == "AAAA"

        tl.consolidate_targets()

        assert len(tl) == 1
        assert len(tl.id_mapping) == 2
        assert tl["AAAA"].objectId == "T101"

        assert "AAAA" not in tl.lookup

        assert tl.id_mapping["T101"] == "T101"
        assert tl.id_mapping["AAAA"] == "T101"

    def test__not_close_not_grouped(self):

        tl = TargetLookup()

        t1 = Target("T101", 45.0, 0.0)
        t2 = Target("T102", 45.0, 2.0)

        tl.add_target(t1)
        tl.add_target(t2)

        tl.consolidate_targets(seplimit=1 * u.deg)

        assert len(tl) == 2

    def test__incomplete_graph_still_merge(self):
        """T1-T2-T3 : if T1 and T2 are 'close', and T2 and T3 are 'close',
        all three should merge even if T1 and T3 are not 'close'"""

        tl = TargetLookup()

        t1 = Target("T101", 44.3, 0.0, alt_ids={"src01": "T101"})
        t2 = Target("T102", 45.0, 0.0, alt_ids={"src02": "T102"})
        t3 = Target("T103", 45.7, 0.0, alt_ids={"src03": "T103"})

        tl.add_target(t1)
        tl.add_target(t2)
        tl.add_target(t3)

        assert t1.coord.separation(t3.coord) > 1 * u.deg
        assert len(tl) == 3

        tl.consolidate_targets(seplimit=1 * u.deg)

        assert len(tl) == 1

        assert set(tl["T101"].alt_ids.values()) == set("T101 T102 T103".split())

    def test__more_complex_merge(self):

        tl = TargetLookup()

        # group 1
        t1 = Target("T101", 45.0, 60.0, alt_ids={"src01": "T101"})
        t2 = Target("SN001", 45.1, 60.1, alt_ids={"src02": "SN001"})  # mod ra/dec
        t3 = Target("OBJ_001", 44.9, 59.9, alt_ids={"src03": "OBJ_001"})

        # add some data to check that it's overwritten
        t1_data = t1.get_target_data("ztf")
        t1_data.meta["parameter"] = 100
        t3_data = t3.get_target_data("ztf")
        t3_data.meta["parameter"] = 10

        # group 2
        t4 = Target("T102", 90.0, 15.0, alt_ids={"src01": "T102"})
        t5 = Target("OBJ_002", 90.1, 15.1, alt_ids={"src03": "OBJ_002"})

        # group 3
        t6 = Target("T103", 180.0, 30.0, alt_ids={"src01": "T103"})

        for target in [t1, t2, t3, t4, t5, t6]:
            tl.add_target(target)

        tl.consolidate_targets(seplimit=1 * u.deg, warn_overwrite=False)

        assert len(tl) == 3
        assert set(tl.keys()) == set("T101 T102 T103".split())
        assert len(tl.id_mapping) == 6
        assert set(tl.id_mapping.keys()) == set(
            "T101 SN001 OBJ_001 T102 OBJ_002 T103".split()
        )

        assert set(tl["T101"].alt_ids.keys()) == set("src01 src02 src03".split())
        assert tl["SN001"].objectId == "T101"
        assert tl["OBJ_001"].objectId == "T101"
        assert tl.id_mapping["T101"] == "T101"
        assert tl.id_mapping["SN001"] == "T101"
        assert tl.id_mapping["OBJ_001"] == "T101"

        assert tl["T101"].target_data["ztf"].meta["parameter"] == 10  # overwritten!

        assert set(tl["T102"].alt_ids.keys()) == set("src01 src03".split())
