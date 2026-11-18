import pytest
import threading
from pathlib import Path
from typing import Callable

from astropy.time import Time

from aas2rto.path_manager import PathManager

from aas2rto.exc import UnknownQueryManagerError
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.primary import PrimaryQueryManager, _qm_task_wrapper
from aas2rto.query_managers.registry import qm_registry  # qm_reg. a SINGLETON
from aas2rto.query_managers.result import QueryManagerResult

from aas2rto.target_lookup import TargetLookup


@pytest.fixture(autouse=True)
def remove_empty_directories(remove_tmp_dirs):
    pass


class RecordingQM(BaseQueryManager):
    def __init__(
        self,
        config: dict,
        target_lookup: TargetLookup,
        parent_path: Path,
        behaviour: Callable = None,
    ):
        super().__init__(config, target_lookup, parent_path)
        self.tasks_performed: bool = False
        self.thread_ident: int = None
        self.behaviour = behaviour  #

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        self.tasks_performed = True
        self.thread_ident = threading.get_ident()
        if self.behaviour:
            self.behaviour()  # Do the extra work

        return QueryManagerResult(name=self.name, success=True)


class MockQM(RecordingQM):
    name = "mock_qm"
    can_run_concurrently = False  # Explicit


class MockConcurrentQM1(RecordingQM):
    name = "cqm1"
    can_run_concurrently = True


class MockConcurrentQM2(RecordingQM):
    name = "cqm2"
    can_run_concurrently = True


class FailingQM(BaseQueryManager):
    name = "failing_qm"

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        try:
            raise ValueError("Requested failure")
        except Exception as e:
            return QueryManagerResult(
                name=self.name, success=False, exc=e, detail="Fail as requested"
            )


class RaisingQM(BaseQueryManager):
    name = "raising_qm"

    def perform_all_tasks(self, iteration, t_ref=None):
        raise ValueError("Unhandled exception")


class Test__InitPrimaryQM:
    def test__no_qms(self, tlookup: TargetLookup, path_mgr: PathManager):
        # Arrange
        empty_config = {}

        # Act
        pqm = PrimaryQueryManager(empty_config, tlookup, path_mgr)

        # Assert
        assert isinstance(pqm.query_managers, dict)
        assert set(pqm.query_managers.keys()) == set()

    def test__all_qms_init(
        self, global_qm_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        pqm = PrimaryQueryManager(global_qm_config, tlookup, path_mgr)

        # Assert
        exp_qms = "atlas fink_lsst fink_ztf tns yse".split()  # lasair_lsst lasair_ztf

        assert set(pqm.query_managers.keys()) == set(exp_qms)

        untested_qms = set(qm_registry.all()) - set(pqm.query_managers.keys())
        if untested_qms:
            msg = (
                f"you should include example configs for the QMs {untested_qms} "
                "in the `global_qm_config` fixture in test_aas2rto/unit/conftest.py "
                "so that they are properly tested"
            )
            raise ValueError(msg)

    def test__respect_use_flag(
        self, atlas_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        config = {"atlas": atlas_config}
        config["atlas"]["use"] = False

        # Act
        pqm = PrimaryQueryManager(config, tlookup, path_mgr)
        # Assert
        assert set(pqm.query_managers.keys()) == set()

    def test__unknown_qm_raises(self, tlookup: TargetLookup, path_mgr: PathManager):
        # Arrange
        config = {"unknown_qm": {}}

        # Act
        with pytest.raises(UnknownQueryManagerError):
            pqm = PrimaryQueryManager(config, tlookup, path_mgr)


class Test__Wrapper:

    def test__run_qm_tasks(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):
        # Arrange
        qm = MockQM({}, tlookup, path_mgr.data_path)

        # Act
        result = _qm_task_wrapper(qm, iteration=1, t_ref=t_fixed)

        # Assert
        assert result.success
        assert not result.exc

    def test__caught_exception_warns(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):
        # Arrange
        qm = FailingQM({}, tlookup, path_mgr.data_path)

        # Act
        with pytest.warns(UserWarning):
            result = _qm_task_wrapper(qm, iteration=1, t_ref=t_fixed)

        # Assert
        assert not result.success
        assert result.detail == "Fail as requested"

    def test__unhandled_exception_warns(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):
        # Arrange
        qm = RaisingQM({}, tlookup, path_mgr.data_path)

        # Act
        result = _qm_task_wrapper(qm, 1, t_fixed)

        # Assert
        assert not result.success
        assert result.detail == "Unhandled exception"


class Test__PerformSeqeuntialAllQMTasks:

    def test__perform(self, tlookup: TargetLookup, path_mgr: PathManager):

        # Arrange
        pqm = PrimaryQueryManager({}, tlookup, path_mgr)
        pqm.query_managers["mock_qm"] = MockQM(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        main_ident = threading.get_ident()

        # Act
        pqm.perform_qm_tasks_sequentially(iteration=1)

        # Assert
        assert pqm.query_managers["mock_qm"].tasks_performed
        assert (
            pqm.query_managers["mock_qm"].thread_ident == main_ident
        )  # Sequential ok!

    def test__caught_exception_warns(
        self, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        pqm = PrimaryQueryManager({}, tlookup, path_mgr)
        pqm.query_managers["mock_qm"] = FailingQM({}, tlookup, path_mgr.data_path)

        # Act
        with pytest.warns(UserWarning):
            pqm.perform_qm_tasks_sequentially(iteration=1)


class Test__PerformQMTasksConcurrent:
    def test__perform_concurrent(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):

        # Arrange
        pqm = PrimaryQueryManager({}, tlookup, path_mgr)
        main_ident = threading.get_ident()
        pqm.query_managers["mock_qm"] = MockQM(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm1"] = MockConcurrentQM1(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm2"] = MockConcurrentQM2(
            {}, tlookup, parent_path=path_mgr.data_path
        )

        # Act
        pqm.perform_qm_tasks_concurrently(1, t_fixed)

        # Assert
        assert pqm.query_managers["mock_qm"].thread_ident == main_ident  # Sequential!
        assert pqm.query_managers["cqm1"].thread_ident != main_ident  # New thread!
        assert pqm.query_managers["cqm2"].thread_ident != main_ident  # New thread!

    def test__barrier_reached(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):
        # Arrange
        barrier = threading.Barrier(2, timeout=2.0)

        passed_barrier = []

        def wait_for_qms():
            barrier.wait()
            passed_barrier.append(threading.get_ident())

        pqm = PrimaryQueryManager({}, tlookup, path_mgr)
        main_ident = threading.get_ident()
        pqm.query_managers["mock_qm"] = MockQM(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm1"] = MockConcurrentQM1(
            {}, tlookup, parent_path=path_mgr.data_path, behaviour=wait_for_qms
        )
        pqm.query_managers["cqm2"] = MockConcurrentQM2(
            {}, tlookup, parent_path=path_mgr.data_path, behaviour=wait_for_qms
        )

        # Act
        pqm.perform_qm_tasks_concurrently(1, t_fixed)

        # Assert
        assert len(passed_barrier) == 2
        assert all([ident != main_ident for ident in passed_barrier])


class Test__ConfigFlagQMTasks:
    def test__config_perform_concurrent(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):

        # Arrange
        config = {
            "primary": {
                "concurrent": True,
            },
        }
        pqm = PrimaryQueryManager(config, tlookup, path_mgr)
        main_ident = threading.get_ident()
        pqm.query_managers["mock_qm"] = MockQM(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm1"] = MockConcurrentQM1(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm2"] = MockConcurrentQM2(
            {}, tlookup, parent_path=path_mgr.data_path
        )

        # Act
        pqm.perform_all_query_manager_tasks(1, t_fixed)

        # Assert
        assert pqm.query_managers["mock_qm"].thread_ident == main_ident  # Sequential!
        assert pqm.query_managers["cqm1"].thread_ident != main_ident  # New thread!
        assert pqm.query_managers["cqm2"].thread_ident != main_ident  # New thread!

    def test__config_perform_sequential(
        self, tlookup: TargetLookup, path_mgr: PathManager, t_fixed: Time
    ):

        # Arrange
        config = {
            "primary": {
                "concurrent": False,
            },
        }
        pqm = PrimaryQueryManager(config, tlookup, path_mgr)
        main_ident = threading.get_ident()
        pqm.query_managers["mock_qm"] = MockQM(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm1"] = MockConcurrentQM1(
            {}, tlookup, parent_path=path_mgr.data_path
        )
        pqm.query_managers["cqm2"] = MockConcurrentQM2(
            {}, tlookup, parent_path=path_mgr.data_path
        )

        # Act
        pqm.perform_all_query_manager_tasks(1, t_fixed)

        # Assert
        assert pqm.query_managers["mock_qm"].tasks_performed
        assert pqm.query_managers["cqm1"].tasks_performed
        assert pqm.query_managers["cqm2"].tasks_performed

        assert pqm.query_managers["mock_qm"].thread_ident == main_ident  # Sequential!
        assert pqm.query_managers["cqm1"].thread_ident == main_ident  # Sequential!
        assert pqm.query_managers["cqm2"].thread_ident == main_ident  # Sequential!
