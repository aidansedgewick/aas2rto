import io
import yaml
from pathlib import Path
from PIL import Image

import pytest

import numpy as np

import telegram

from dk154_targets.messengers import telegram_messenger
from dk154_targets.messengers.telegram_messenger import TelegramMessenger
from dk154_targets.exc import UnexpectedKeysWarning, MissingMediaWarning


@pytest.fixture
def config():
    return {
        "token": "12345",
        "users": {1001: "user_001", 1002: "user_002"},
        "sudoers": {1003: "user_003"},
    }


@pytest.fixture
def extra_users():
    return {1004: "user_004", 1005: "user_005"}


def save_image_to_path(filepath, size=(10, 10)):
    x = np.random.uniform(0, 1, size) * 256
    im = Image.fromarray(np.uint8(x))
    im.save(filepath)


class MockBot:

    def __init__(self, token):
        self.token = token

        self.messages_sent = []
        self.media_sent = []

    def send_message(self, chat_id=None, text=None, disable_web_page_preview=True):
        if chat_id is None:
            raise ValueError("chat_id is None.")
        if text is None:
            raise ValueError("text is None")
        self.messages_sent.append((chat_id, text))
        return (chat_id, text)

    def send_photo(self, chat_id, photo=None, caption=None):
        if chat_id is None:
            raise ValueError("chat_id is None.")
        if photo is None:
            raise ValueError("photo is None")
        self.media_sent.append((chat_id, photo))
        return (chat_id, photo)

    def send_media_group(self, chat_id=None, media=None):
        if isinstance(media, Path):
            raise ValueError("send_media_group should have media=<list-of-media>")

        if chat_id is None:
            raise ValueError("chat_id is None.")
        if media is None:
            raise ValueError("media is None")

        media_sent = [(chat_id, media_ii) for media_ii in media]
        self.media_sent.extend(media_sent)
        return media_sent


@pytest.fixture(autouse=True)
def mock_bot(monkeypatch):
    monkeypatch.setattr(
        "dk154_targets.messengers.telegram_messenger.telegram.Bot", MockBot
    )


class Test__TelegramMessengerInit:

    def test__empty_config_has_nothing(self):

        msgr = TelegramMessenger({})

        assert msgr.token is None
        assert isinstance(msgr.users, dict)
        assert len(msgr.users) == 0
        assert isinstance(msgr.sudoers, dict)
        assert len(msgr.users) == 0

    def test__normal_config(self, config):

        msgr = TelegramMessenger(config)

        assert msgr.token == "12345"

        assert isinstance(msgr.users, dict)
        assert set(msgr.users.keys()) == set([1001, 1002, 1003])  # user 003 added!
        assert msgr.users[1001] == "user_001"
        assert msgr.users[1002] == "user_002"
        assert msgr.users[1003] == "user_003"

        assert isinstance(msgr.sudoers, dict)
        assert set(msgr.sudoers.keys()) == set([1003])
        assert msgr.sudoers[1003] == "user_003"

        assert msgr.users_file is None

        assert msgr.bot is not None

    def test__warn_on_unexpected_config_key(self, config):
        config["unexp_kw"] = 10

        with pytest.warns(UnexpectedKeysWarning):
            msgr = TelegramMessenger(config)

    def test__users_as_list(self, config):
        config["users"] = [1001, 1002]

        msgr = TelegramMessenger(config)

        assert isinstance(msgr.users, dict)
        assert set(msgr.users.keys()) == set([1001, 1002, 1003])  # user 003 added!
        assert msgr.users[1001] == "unknown_user"
        assert msgr.users[1002] == "unknown_user"
        assert msgr.users[1003] == "user_003"

        assert msgr.bot is not None

    def test__sudoers_as_list(self, config):
        config["sudoers"] = [1003]

        msgr = TelegramMessenger(config)

        assert isinstance(msgr.sudoers, dict)
        assert set(msgr.users.keys()) == set([1001, 1002, 1003])  # user 003 added!
        assert msgr.users[1001] == "user_001"
        assert msgr.users[1002] == "user_002"
        assert msgr.users[1003] == "unknown_sudoer"

        assert isinstance(msgr.sudoers, dict)
        assert msgr.sudoers[1003] == "unknown_sudoer"

        assert msgr.bot is not None

    def test__init_with_user_file(self, config, tmp_path):
        config["users_file"] = str(tmp_path)

        msgr = TelegramMessenger(config)

        assert isinstance(msgr.users_file, Path)

    def test__init_telegram_module_is_none(self, config, monkeypatch):

        assert telegram_messenger.telegram is not None

        with monkeypatch.context() as m:
            m.setattr("dk154_targets.messengers.telegram_messenger.telegram", None)
            assert telegram_messenger.telegram is None

            msgr = TelegramMessenger(config)

            assert msgr.token is not None
            assert msgr.bot is None

        assert telegram_messenger.telegram is not None


class Test__ReadNewUsers:

    def test__new_users_is_empty(self, config, tmp_path):
        users_file = tmp_path / "users.yaml"
        config["users_file"] = users_file

        msgr = TelegramMessenger(config)

        result = msgr.read_new_users()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test__read_new_users(self, config: dict, extra_users: dict, tmp_path):
        users_file = tmp_path / "users.yaml"
        config["users_file"] = users_file

        with open(users_file, "w+") as f:
            yaml.dump(extra_users, f)

        msgr = TelegramMessenger(config)
        result = msgr.read_new_users()

        assert set(result.keys()) == set([1004, 1005])
        assert result[1004] == "user_004"
        assert result[1005] == "user_005"

    def test__read_new_users_missing_names(
        self, config: dict, extra_users: dict, tmp_path
    ):
        users_file = tmp_path / "users.yaml"
        config["users_file"] = users_file

        with open(users_file, "w+") as f:
            yaml.dump(list(extra_users.keys()), f)

        msgr = TelegramMessenger(config)
        result = msgr.read_new_users()

        assert set(result.keys()) == set([1004, 1005])
        assert result[1004] == "unknown_user"
        assert result[1005] == "unknown_user"


class Test__SendTextMessages:

    def test__send_single_message_to_single_user(self, config):

        msgr = TelegramMessenger(config)

        user_id = 101
        message = "a test message"

        sent_messages = msgr.send_to_user(user_id, texts=message)

        assert isinstance(msgr.bot, MockBot)  # so we can track the messages sent!

        assert len(msgr.bot.messages_sent) == 1

        first_message = msgr.bot.messages_sent[0]
        assert first_message[0] == 101
        assert first_message[1] == "a test message"

    def test__send_many_messages_to_single_user(self, config):

        msgr = TelegramMessenger(config)

        user_id = 101
        msg_01 = "a test message"
        msg_02 = "another message"

        msgr.send_to_user(user_id, texts=[msg_01, msg_02])

        assert isinstance(msgr.bot, MockBot)

        assert len(msgr.bot.messages_sent) == 2

        first_msg = msgr.bot.messages_sent[0]
        assert first_msg[0] == 101
        assert first_msg[1] == "a test message"
        second_msg = msgr.bot.messages_sent[1]
        assert second_msg[0] == 101
        assert second_msg[1] == "another message"

    def test__long_messages_are_chunked(self, config):

        msg = "ABCDEFGHIJKLM"

        msgr = TelegramMessenger(config)

        msgr.CHUNK_MAX = 5
        assert TelegramMessenger.CHUNK_MAX == 4000  # class not changed...

        sent_messages = msgr.send_to_user(101, texts=msg)

        assert len(sent_messages) == 3
        assert sent_messages[0][1] == "ABCDE"
        assert sent_messages[1][1] == "FGHIJ"
        assert sent_messages[2][1] == "KLM"

        assert len(msgr.bot.messages_sent) == 3
        assert msgr.bot.messages_sent[0][1] == "ABCDE"
        assert msgr.bot.messages_sent[1][1] == "FGHIJ"
        assert msgr.bot.messages_sent[2][1] == "KLM"


class Test__SendImages:

    def test__send_single_image_to_single_user(self, config, tmp_path):

        filepath = tmp_path / "testim.png"
        save_image_to_path(filepath)
        assert filepath.exists()

        msgr = TelegramMessenger(config)
        sent_messages = msgr.send_to_user(101, img_paths=filepath)

        assert isinstance(msgr.bot, MockBot)

        assert len(sent_messages) == 1
        assert sent_messages[0][0] == 101
        assert isinstance(sent_messages[0][1], bytes)
        # assert sent_messages[0][1].name == str(filepath)

    def test__send_many_images_to_single_user(self, config, tmp_path):

        filepath_01 = tmp_path / "testim_01.png"
        save_image_to_path(filepath_01)
        assert filepath_01.exists()

        filepath_02 = tmp_path / "testim_02.png"
        save_image_to_path(filepath_02)
        assert filepath_02.exists()

        msgr = TelegramMessenger(config)

        sent_messages = msgr.send_to_user(101, img_paths=[filepath_01, filepath_02])

        assert len(sent_messages) == 2
        assert sent_messages[0][0] == 101
        assert isinstance(sent_messages[0][1], telegram.InputMediaPhoto)
        assert sent_messages[0][1].media.filename == str(filepath_01)
        assert sent_messages[1][0] == 101
        assert isinstance(sent_messages[1][1], telegram.InputMediaPhoto)
        assert sent_messages[1][1].media.filename == str(filepath_02)

    def test__missing_image_does_not_fail_send_to_user(self, config, tmp_path):

        filepath_01 = tmp_path / "testim_01.png"
        save_image_to_path(filepath_01)
        assert filepath_01.exists()

        filepath_02 = tmp_path / "testim_02.png"
        # DON'T save the file here!
        assert not filepath_02.exists()

        filepath_03 = tmp_path / "testim_03.png"
        save_image_to_path(filepath_03)
        assert filepath_03.exists()

        msgr = TelegramMessenger(config)

        img_paths = [filepath_01, filepath_02, filepath_03]
        with pytest.warns(MissingMediaWarning):
            sent_messages = msgr.send_to_user(101, img_paths=img_paths)

        assert len(sent_messages) == 2
        assert sent_messages[0][0] == 101
        assert isinstance(sent_messages[0][1], telegram.InputMediaPhoto)
        assert sent_messages[0][1].media.filename == str(filepath_01)
        assert isinstance(sent_messages[1][1], telegram.InputMediaPhoto)
        assert sent_messages[1][1].media.filename == str(filepath_03)


class Test__MessageUsers:

    def test__send_text_to_default_users(self, config):

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(texts="sample message")

        assert len(sent_messages) == 3
        assert sent_messages[0][0] == 1001
        assert sent_messages[0][1] == "sample message"
        assert sent_messages[1][0] == 1002
        assert sent_messages[1][1] == "sample message"
        assert sent_messages[2][0] == 1003
        assert sent_messages[2][1] == "sample message"

    def test__send_text_to_single_users(self, config):

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(
            users=101, texts="sample message"
        )

        assert len(sent_messages) == 1
        assert sent_messages[0][0] == 101
        assert sent_messages[0][1] == "sample message"

    def test__send_text_to_many_users(self, config):

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(
            users=[101, 102], texts="sample message"
        )

        assert len(sent_messages) == 2
        assert sent_messages[0][0] == 101
        assert sent_messages[0][1] == "sample message"
        assert sent_messages[1][0] == 102
        assert sent_messages[1][1] == "sample message"

    def test__sent_text_to_sudoers(self, config):

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(
            users="sudoers", texts="sample message"
        )

        assert len(sent_messages) == 1
        assert sent_messages[0][0] == 1003
        assert sent_messages[0][1] == "sample message"

    def test__send_text_to_extra_users(self, config, extra_users, tmp_path):
        users_file = tmp_path / "users.yaml"
        config["users_file"] = users_file

        with open(users_file, "w+") as f:
            yaml.dump(list(extra_users.keys()), f)

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(texts="sample message")
        assert len(sent_messages) == 5

        assert sent_messages[0][0] == 1001
        assert sent_messages[0][1] == "sample message"
        assert sent_messages[1][0] == 1002
        assert sent_messages[1][1] == "sample message"
        assert sent_messages[2][0] == 1003
        assert sent_messages[2][1] == "sample message"
        assert sent_messages[3][0] == 1004
        assert sent_messages[3][1] == "sample message"
        assert sent_messages[4][0] == 1005
        assert sent_messages[4][1] == "sample message"

    def send__single_photo(self, config, tmp_path):

        filepath_01 = tmp_path / "testim_01.png"
        save_image_to_path(filepath_01)
        assert filepath_01.exists()

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(img_paths=filepath_01)

        assert len(sent_messages) == 3
        assert sent_messages[0][0] == 1001
        assert isinstance(sent_messages[0][1], bytes)
        assert sent_messages[1][0] == 1002
        assert isinstance(sent_messages[1][1], bytes)
        assert sent_messages[2][0] == 1003
        assert isinstance(sent_messages[2][1], bytes)

    def test__send_many_photos(self, config, tmp_path):

        filepath_01 = tmp_path / "testim_01.png"
        save_image_to_path(filepath_01)
        assert filepath_01.exists()

        filepath_02 = tmp_path / "testim_02.png"
        save_image_to_path(filepath_02)
        assert filepath_02.exists()

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(
            img_paths=[filepath_01, filepath_02]
        )

        assert len(sent_messages) == 6

        assert sent_messages[0][0] == 1001
        assert isinstance(sent_messages[0][1], telegram.InputMediaPhoto)
        assert sent_messages[0][1].media.filename == str(filepath_01)

        assert sent_messages[1][0] == 1001
        assert isinstance(sent_messages[1][1], telegram.InputMediaPhoto)
        assert sent_messages[1][1].media.filename == str(filepath_02)

        assert sent_messages[2][0] == 1002
        assert isinstance(sent_messages[2][1], telegram.InputMediaPhoto)
        assert sent_messages[2][1].media.filename == str(filepath_01)

        assert sent_messages[3][0] == 1002
        assert isinstance(sent_messages[3][1], telegram.InputMediaPhoto)
        assert sent_messages[3][1].media.filename == str(filepath_02)

        assert sent_messages[4][0] == 1003
        assert isinstance(sent_messages[4][1], telegram.InputMediaPhoto)
        assert sent_messages[4][1].media.filename == str(filepath_01)

        assert sent_messages[5][0] == 1003
        assert isinstance(sent_messages[5][1], telegram.InputMediaPhoto)
        assert sent_messages[5][1].media.filename == str(filepath_02)

    def test__missing_photo_not_sent(self, config, tmp_path):
        filepath_01 = tmp_path / "testim_01.png"
        save_image_to_path(filepath_01)
        assert filepath_01.exists()

        filepath_02 = tmp_path / "testim_02.png"
        # intentionally not saved!
        assert not filepath_02.exists()

        filepath_03 = tmp_path / "testim_03.png"
        save_image_to_path(filepath_03)
        assert filepath_03.exists()

        msgr = TelegramMessenger(config)

        sent_messages, exceptions = msgr.message_users(
            img_paths=[filepath_01, filepath_02, filepath_03]
        )

        assert len(exceptions) == 1
        assert "testim_02.png missing!" in exceptions[0]

        # ...everything else should behave as normal.
        assert len(sent_messages) == 6

        assert sent_messages[0][0] == 1001
        assert isinstance(sent_messages[0][1], telegram.InputMediaPhoto)
        assert sent_messages[0][1].media.filename == str(filepath_01)

        assert sent_messages[1][0] == 1001
        assert isinstance(sent_messages[1][1], telegram.InputMediaPhoto)
        assert sent_messages[1][1].media.filename == str(filepath_03)

        assert sent_messages[2][0] == 1002
        assert isinstance(sent_messages[2][1], telegram.InputMediaPhoto)
        assert sent_messages[2][1].media.filename == str(filepath_01)

        assert sent_messages[3][0] == 1002
        assert isinstance(sent_messages[3][1], telegram.InputMediaPhoto)
        assert sent_messages[3][1].media.filename == str(filepath_03)

        assert sent_messages[4][0] == 1003
        assert isinstance(sent_messages[4][1], telegram.InputMediaPhoto)
        assert sent_messages[4][1].media.filename == str(filepath_01)

        assert sent_messages[5][0] == 1003
        assert isinstance(sent_messages[5][1], telegram.InputMediaPhoto)
        assert sent_messages[5][1].media.filename == str(filepath_03)
