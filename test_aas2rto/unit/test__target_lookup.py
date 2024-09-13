import pytest

from aas2rto.exc import NotATargetError
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def mock_target():
    return Target("ZTF00abc", 30.0, 45.0)


@pytest.fixture
def mock_target_alt_ids():
    alt_names = {"tns": "SN2001A"}
    return Target("ZTF01def", 60.0, -30.0, alternative_ids=alt_names)


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

    def test__dunder_set_fails_on_not_target(self):
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

        assert "ZTF01def" in tlookup
        assert "SN2001A" in tlookup

        removed = tlookup.pop("ZTF01def")
        assert len(tlookup) == 1
        assert "ZTF01def" not in tlookup
        assert "SN2001A" not in tlookup

        assert removed.objectId == "ZTF01def"

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
        new_targ = Target("ZTF02ijk", 45.0, -45.0, alternative_ids=alt_ids)

        tlookup = TargetLookup()

        tlookup.add_target(new_targ)

        assert len(tlookup) == 1
        assert "ZTF02ijk" in tlookup
        assert "02xyz" in tlookup
        assert "AT_001" in tlookup

        assert "datasrc" not in tlookup
        assert "othersrc" not in tlookup
