import pytest

import numpy as np

from astropy.time import Time

from aas2rto.messaging.messaging_manager import MessagingManager, DefaultMessageFilter
from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.messaging.slack_messenger import SlackMessenger
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def msg_filter():
    return DefaultMessageFilter()


class Test__DefaultMessageFilter:
    def test__init(self):
        # Act
        f = DefaultMessageFilter()

    def test__allow(
        self, basic_target: Target, msg_filter: DefaultMessageFilter, t_fixed: Time
    ):
        # Arrange
        basic_target.updated = True
        basic_target.info_messages.append("Some information")
        basic_target.update_science_score_history(1.0, t_ref=t_fixed)

        # Act
        allow, reasons = msg_filter(basic_target, t_ref=t_fixed)

        # Assert
        assert allow
        assert set(reasons) == set()

    def test__nothing(
        self, basic_target: Target, msg_filter: DefaultMessageFilter, t_fixed: Time
    ):
        # Act
        allow, reasons = msg_filter(basic_target, t_ref=t_fixed)

        # Assert
        assert not allow
        assert set(reasons) == set(["no science score", "no messages", "not updated"])

    def test__no_score(
        self, basic_target: Target, msg_filter: DefaultMessageFilter, t_fixed: Time
    ):
        # Arrange
        basic_target.updated = True
        basic_target.info_messages.append("Some information")

        # Act
        allow, reasons = msg_filter(basic_target, t_ref=t_fixed)

        # Assert
        assert not allow
        assert set(reasons) == set(["no science score"])

    def test__low_score(
        self, basic_target: Target, msg_filter: DefaultMessageFilter, t_fixed: Time
    ):
        # Arrange
        basic_target.updated = True
        basic_target.info_messages.append("Some information")
        basic_target.update_science_score_history(-1.0, t_ref=t_fixed)

        # Act
        allow, reasons = msg_filter(basic_target, t_ref=t_fixed)

        # Assert
        assert not allow
        assert set(reasons) == set(["low score"])

    def test__no_recent_sent(
        self, basic_target: Target, msg_filter: DefaultMessageFilter, t_fixed: Time
    ):
        # Arrange
        basic_target.updated = True
        basic_target.info_messages.append("Some information")
        basic_target.update_science_score_history(1.0, t_ref=t_fixed)
        first_allow, reasons = msg_filter(basic_target, t_ref=t_fixed)
        assert first_allow  # reminder
        basic_target.last_message_time = t_fixed

        # Act
        allow, reasons = msg_filter(basic_target, t_ref=t_fixed)

        # Assert
        assert not allow
        assert set(reasons) == set(["recent message"])


class Test__MsgMgrInit:
    def test__normal_init(
        self, msg_mgr_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        mgr = MessagingManager(msg_mgr_config, tlookup, path_mgr)

        # Assert
        assert isinstance(mgr.telegram_messenger, TelegramMessenger)
        assert isinstance(mgr.slack_messenger, SlackMessenger)

    def test__no_use_respected(
        self, msg_mgr_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        msg_mgr_config["telegram"]["use"] = False
        msg_mgr_config["slack"]["use"] = False

        # Act
        mgr = MessagingManager(msg_mgr_config, tlookup, path_mgr)

        # Assert
        assert mgr.telegram_messenger is None
        assert mgr.slack_messenger is None

    def test__unknown_mgr_raises(
        self, msg_mgr_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        msg_mgr_config["other_msgr"] = {"use": True, "token": "xyz", "aaagh": 10}

        # Act
        with pytest.raises(NotImplementedError):
            mgr = MessagingManager(msg_mgr_config, tlookup, path_mgr)


class Test__SudoMessages:
    def test__message_sudoers(self, msg_mgr: MessagingManager):
        # Act
        sent, exc = msg_mgr.send_sudo_messages(texts="some message")

        # Assert
        assert len(sent) == 1
        assert sent[0]["user"] == 901

    def test__no_tele_msgr(self, msg_mgr: MessagingManager):
        # Arrange
        msg_mgr.telegram_messenger = None

        # Act
        sent, exc = msg_mgr.send_sudo_messages(texts="blah")

        # Assert
        assert len(sent) == 0
        assert len(exc) == 0


class Test__MessagingTasks:
    def test__updated_and_scored(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        T00 = msg_mgr.target_lookup["T00"]
        T00.update_science_score_history(1.0, t_fixed)
        T00.updated = True
        T00.info_messages.append("here's a message!")
        assert T00.last_message_time is None  # reminder...

        # Act
        results, reasons = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(results["sent"]) == set(["T00"])
        assert set(results["skipped"]) == set(["T01"])

        assert isinstance(T00.last_message_time, Time)
        assert np.isclose(T00.last_message_time.mjd - 60000.0, 0.0)

        assert "T01" in reasons["not updated"]
        assert "T01" in reasons["no messages"]
        assert "T01" in reasons["no science score"]

    def test__no_update_msgs(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        msg_mgr.target_lookup["T00"].update_science_score_history(1.0, t_fixed)
        msg_mgr.target_lookup["T00"].updated = True
        # DON'T attach update messages

        # Act
        results, reasons = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(results.keys()) == set(["sent", "skipped", "error"])
        assert set(results["sent"]) == set()
        assert set(results["skipped"]) == set(["T00", "T01"])

        # check filter 'reason' names on fail
        assert "T00" in reasons["no messages"]
        assert "T01" in reasons["not updated"]

    def test__no_score(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        # DON'T update score.
        msg_mgr.target_lookup["T00"].updated = True
        msg_mgr.target_lookup["T00"].info_messages.append("here's a message!")

        # Act
        results, reasons = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(results.keys()) == set(["sent", "skipped", "error"])
        assert set(results["sent"]) == set()
        assert set(results["skipped"]) == set(["T00", "T01"])

    def test__bad_score(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        msg_mgr.target_lookup["T00"].update_science_score_history(-1.0, t_fixed)
        msg_mgr.target_lookup["T00"].updated = True
        msg_mgr.target_lookup["T00"].info_messages.append("here's a message!")

        # Act
        results, reasons = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(results.keys()) == set(["sent", "skipped", "error"])
        assert set(results["sent"]) == set()
        assert set(results["skipped"]) == set(["T00", "T01"])


class Test__CrashReports:
    def test__crash_reports(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Act
        sent, exc = msg_mgr.send_crash_reports(text="report: crash in program...")

        # Assert
        assert len(sent) == 2
        assert sent[0]["msg_type"] == "text"

    def test__no_msgr(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        msg_mgr.telegram_messenger = None

        # Act
        sent, exc = msg_mgr.send_crash_reports(text="some unsent report...")

        # Assert
        assert len(sent) == 0
        assert len(exc) == 0
