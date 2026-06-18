from __future__ import annotations

import pytest

from aas2rto.exc import AlreadyRegisteredError
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.registry import QueryManagerRegistry, qm_registry


@pytest.fixture
def mock_registry():
    return QueryManagerRegistry()


class MockQM(BaseQueryManager):
    name = "mock_qm"

    def perform_all_tasks(self, iteration, t_ref=None):
        pass


class Test__QMRegistrySingleton:
    def test__singleton_instance(self):
        # Assert
        assert isinstance(qm_registry, QueryManagerRegistry)

    def test__singleton_has_some_qms_registered(self):
        # Assert
        assert len(qm_registry.all()) > 0


class Test__RegisteringNew:
    def test__new_registry_empty(self):
        # Act
        qmr = QueryManagerRegistry()

        # Assert
        assert set(qmr.all()) == set()  # Nothing!

    def test__register_new_class_decorator(self, mock_registry: QueryManagerRegistry):
        # Act

        # Only test decorator here, elsewhere use (private) inline syntax
        @mock_registry.register()
        class DecMockQM(BaseQueryManager):
            name = "mock_qm"

            def perform_all_tasks(self, iteration, t_ref=None):
                pass

        # Assert
        assert set(mock_registry.all()) == set(["mock_qm"])

    def test__register_inline(self, mock_registry: QueryManagerRegistry):
        # Act
        mock_registry._register_class(MockQM)

        # Assert
        assert set(mock_registry.all()) == set(["mock_qm"])

    def test__reregister_same_no_raises(self, mock_registry: QueryManagerRegistry):
        # Arrange
        mock_registry._register_class(MockQM)

        # Act
        mock_registry._register_class(MockQM)

    def test__rereister_new_raises(self, mock_registry: QueryManagerRegistry):
        # Arrange
        mock_registry._register_class(MockQM)

        class OtherMockQM(BaseQueryManager):
            name = "mock_qm"  # The same as before!

            def perform_all_tasks(self, iteration, t_ref=None):
                pass

        assert MockQM is not OtherMockQM

        # Act
        with pytest.raises(AlreadyRegisteredError):
            mock_registry._register_class(OtherMockQM)
