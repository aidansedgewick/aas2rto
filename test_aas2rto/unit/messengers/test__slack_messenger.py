import pytest
from pathlib import Path

import astropy.time as Time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from matplotlib import pyplot as plt

from aas2rto.exc import UnexpectedKeysWarning, MissingRequiredConfigKeyError
from aas2rto.messaging.slack_messenger import SlackMessenger
from aas2rto.target import Target


def write_image(filepath: Path):
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.savefig(filepath)
    plt.close(fig)


class Test__SlackMsgrInit:
    def test__msgr_init(self, slack_config: dict):
        # Act
        msgr = SlackMessenger(slack_config)

        # Assert
        assert isinstance(msgr.client, WebClient)

    def test__unexpected_key_warns(self, slack_config: dict):
        # Arrange
        slack_config["blah"] = 100.0

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            msgr = SlackMessenger(slack_config)

        # Assert
        assert isinstance(msgr.client, WebClient)

    def test__missing_token_raises(self, slack_config: dict):
        # Arrange
        slack_config.pop("token")

        # Act
        with pytest.raises(MissingRequiredConfigKeyError):
            msgr = SlackMessenger(slack_config)

    def test__missing_channel_id_raises(self, slack_config: dict):
        # Arrange
        slack_config.pop("channel_id")

        # Act
        with pytest.raises(MissingRequiredConfigKeyError):
            msgr = SlackMessenger(slack_config)

    def test__no_slack_sdk_raises(self, slack_config: dict, monkeypatch):
        # Act
        with monkeypatch.context() as mc:
            mc.setattr("aas2rto.messaging.slack_messenger.slack_sdk", None)
            with pytest.raises(ModuleNotFoundError):
                msgr = SlackMessenger(slack_config)


class Test__SendMessages:
    def test__send_str_text(self, slack_msgr: SlackMessenger):
        # Arrange
        txt = "some message"

        # Act
        slack_msgr.send_messages(texts=txt)

    def test__send_text_list(self, slack_msgr: SlackMessenger):
        # Arrange
        txt_list = ["message1", "message2"]

        # Act
        slack_msgr.send_messages(texts=txt_list)

    def test__send_text_none_no_fail(self, slack_msgr: SlackMessenger):
        # Act
        slack_msgr.send_messages()

    def test__send_file(self, slack_msgr: SlackMessenger, tmp_path: Path):
        # Arrange
        filepath = tmp_path / "img01.png"
        write_image(filepath)

        # Act
        slack_msgr.send_messages(img_paths=filepath, comment="comment")

    def test__send_file_list(self, slack_msgr: SlackMessenger, tmp_path: Path):
        # Arrange
        filepath01 = tmp_path / "img01.png"
        write_image(filepath01)
        filepath02 = tmp_path / "img02.png"
        write_image(filepath02)

        # Act
        slack_msgr.send_messages(img_paths=[filepath01, filepath02], comment="comment")

    def test__client_none_no_raise(self, slack_msgr: SlackMessenger):
        # Arrange
        slack_msgr.client = None

        # Act
        slack_msgr.send_messages()


class Test__ApiErrors:
    def test__text_no_raise(self, raising_slack_msgr: SlackMessenger):
        # Act
        raising_slack_msgr.send_messages(texts="text")

    def test__img_file_no_raise(
        self, raising_slack_msgr: SlackMessenger, tmp_path: Path
    ):
        # Arrange
        filepath = tmp_path / "img01.png"
        write_image(filepath)

        # Act
        raising_slack_msgr.send_messages(img_paths=filepath)
