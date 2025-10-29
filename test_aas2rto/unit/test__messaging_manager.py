import pytest

from astropy.time import Time

from aas2rto.messaging.messaging_manager import MessagingManager
from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.messaging.slack_messenger import SlackMessenger
from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup


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
        msg_mgr.target_lookup["T00"].update_science_score_history(1.0, t_fixed)
        msg_mgr.target_lookup["T00"].updated = True
        msg_mgr.target_lookup["T00"].update_messages.append("here's a message!")

        # Act
        sent, skipped, no_updates = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(sent) == set(["T00"])
        assert set(skipped) == set(["T01"])

    def test__no_update_msgs(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        msg_mgr.target_lookup["T00"].update_science_score_history(1.0, t_fixed)
        msg_mgr.target_lookup["T00"].updated = True
        # DON'T attach update messages

        # Act
        sent, skipped, no_updates = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(sent) == set()
        assert set(skipped) == set(["T01"])
        assert set(no_updates) == set(["T00"])

    def test__no_score(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        # DON'T update score.
        msg_mgr.target_lookup["T00"].updated = True
        msg_mgr.target_lookup["T00"].update_messages.append("here's a message!")

        # Act
        sent, skipped, no_updates = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(sent) == set()
        assert set(skipped) == set(["T00", "T01"])
        assert set(no_updates) == set()

    def test__bad_score(self, msg_mgr: MessagingManager, t_fixed: Time):
        # Arrange
        msg_mgr.target_lookup["T00"].update_science_score_history(-1.0, t_fixed)
        msg_mgr.target_lookup["T00"].updated = True
        msg_mgr.target_lookup["T00"].update_messages.append("here's a message!")

        # Act
        sent, skipped, no_updates = msg_mgr.perform_messaging_tasks(t_ref=t_fixed)

        # Assert
        assert set(sent) == set()
        assert set(skipped) == set(["T00", "T01"])
        assert set(no_updates) == set()


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
