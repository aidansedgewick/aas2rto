import pickle
import pytest
import string
import yaml
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import telegram

from aas2rto.messaging.telegram_messenger import TelegramMessenger


def write_image(filepath: Path):
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.savefig(filepath)
    plt.close(fig)


class Test__TelegramInit:
    def test__telegram_init(self, telegram_config: dict):
        # Act
        msgr = TelegramMessenger(telegram_config)

        # Assert
        assert set(msgr.users.keys()) == set([101, 102, 901])
        assert set(msgr.sudoers.keys()) == set([901])

    def test__users_is_int(self, telegram_config: dict):
        # Arrange
        telegram_config["users"] = 201
        telegram_config["sudoers"] = 801

        # Act
        msgr = TelegramMessenger(telegram_config)

        # Assert
        assert set(msgr.users.keys()) == set([201, 801])
        assert msgr.users[201] == "unknown_user"
        assert msgr.users[801] == "unknown_sudoer"

        assert set(msgr.sudoers.keys()) == set([801])
        assert msgr.sudoers[801] == "unknown_sudoer"


class Test__GetBot:
    def test__get_bot(self, telegram_config: dict):
        # Arrange
        msgr = TelegramMessenger(telegram_config)

        # Act
        bot = msgr.get_bot()

        # Assert
        assert isinstance(bot, telegram.Bot)


class Test__ReadUsers:
    def test__file_none_no_fail(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        assert "users_file" not in telegram_config
        msgr = TelegramMessenger(telegram_config)

        # Act
        new_users = msgr.read_users_file()

        # Assert
        assert isinstance(new_users, dict)
        assert len(new_users) == 0

    def test__missing_file_no_fail(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        telegram_config["users_file"] = tmp_path / "new_users.yaml"
        msgr = TelegramMessenger(telegram_config)

        # Act
        new_users = msgr.read_users_file()

        # Assert
        assert isinstance(new_users, dict)
        assert len(new_users) == 0

    def test__read_new_users(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        new_users = {301: "new_user301", 302: "new_user302"}
        users_path = tmp_path / "new_users.yaml"
        with open(users_path, "w+") as f:
            yaml.dump(new_users, f)

        telegram_config["users_file"] = users_path
        msgr = TelegramMessenger(telegram_config)

        # Act
        new_users = msgr.read_users_file()

        # Assert
        assert set(new_users.keys()) == set([301, 302])

    def test__read_empty_file(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        users_path = tmp_path / "new_users.yaml"
        users_path.touch()
        telegram_config["users_file"] = users_path
        msgr = TelegramMessenger(telegram_config)

        # Act
        new_users = msgr.read_users_file()

        # Assert
        assert isinstance(new_users, dict)
        assert len(new_users) == 0


class Test__GetUsers:
    def test__no_users_file(self, telegram_config: dict):
        # Arrange
        msgr = TelegramMessenger(telegram_config)

        # Act
        users = msgr.get_users()

        # Assert
        assert set(users.keys()) == set([101, 102, 901])

    def test__with_users_file(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        new_users = {301: "new_user301", 302: "new_user302"}
        users_path = tmp_path / "new_users.yaml"
        with open(users_path, "w+") as f:
            yaml.dump(new_users, f)

        telegram_config["users_file"] = users_path
        msgr = TelegramMessenger(telegram_config)

        # Act
        users = msgr.get_users()

        # Assert
        assert set(users.keys()) == set([101, 102, 901, 301, 302])

    def test__malformed_file_caught(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        users_path = tmp_path / "new_users.yaml"
        with open(users_path, "wb+") as f:
            pickle.dump(np.arange(100), f)

        telegram_config["users_file"] = users_path
        msgr = TelegramMessenger(telegram_config)

        # Act
        with pytest.warns(UserWarning):
            users = msgr.get_users()

        # Assert
        assert set(users.keys()) == set([101, 102, 901])

    def test__empty_file(self, telegram_config: dict, tmp_path: Path):
        # Arrange
        users_path = tmp_path / "new_users.yaml"
        users_path.touch()

        telegram_config["users_file"] = users_path
        msgr = TelegramMessenger(telegram_config)

        # Act
        users = msgr.get_users()

        # Assert
        assert set(users.keys()) == set([101, 102, 901])


class Test__SendToUser:
    def test__send_single_text(self, telegram_msgr: TelegramMessenger):
        # Act
        sent = telegram_msgr.send_to_user(201, texts="some_message")
        # NOTE: this is with the monkeypatched object!

        # Assert
        assert len(sent) == 1
        assert sent[0]["msg_type"] == "text"
        assert sent[0]["user"] == 201

    def test__send_long_message(self, telegram_msgr: TelegramMessenger):
        # Arrange
        chars = list(string.ascii_lowercase) + [" "]
        msg = "".join(np.random.choice(chars, 4500))

        # Act
        sent = telegram_msgr.send_to_user(user=201, texts=msg)

        # Assert
        assert len(sent) == 2
        assert sent[0]["length"] == 4000
        assert sent[1]["length"] == 500

    def test__send_single_photo(self, telegram_msgr: TelegramMessenger, tmp_path: Path):
        # Arrange
        img_path = tmp_path / "img001.png"
        write_image(img_path)

        # Act
        sent = telegram_msgr.send_to_user(201, img_paths=img_path)
        # NOTE: this is with the monkeypatched object!

        # Assert
        assert len(sent) == 1
        assert sent[0]["msg_type"] == "single_photo"

    def test__send_media_group(self, telegram_msgr: TelegramMessenger, tmp_path: Path):
        # Arrange
        img_paths = []
        for ii in range(11):
            img_path = tmp_path / f"img{ii:03d}.png"
            write_image(img_path)
            img_paths.append(img_path)

        # Act
        sent = telegram_msgr.send_to_user(201, img_paths=img_paths)

        # Assert
        assert len(sent) == 2
        assert sent[0]["length"] == 10
        assert sent[0]["msg_type"] == "media_group"
        assert sent[1]["length"] == "n/a"
        assert sent[1]["msg_type"] == "single_photo"


class Test__MessageUsers:
    def test__single_text(self, telegram_msgr: TelegramMessenger):
        # Act
        sent, exc = telegram_msgr.message_users(texts="a message")

        # Assert
        assert len(sent) == 3
        assert sent[0]["user"] == 101
        assert sent[1]["user"] == 102
        assert sent[2]["user"] == 901

    def test__send_single_photo(self, telegram_msgr: TelegramMessenger, tmp_path: Path):
        # Arrange
        img_path = tmp_path / "img001.png"
        write_image(img_path)

        # Act
        sent, exc = telegram_msgr.message_users(img_paths=img_path)

        # Arrange
        assert len(sent) == 3
        assert sent[0]["msg_type"] == "single_photo"
        assert sent[1]["msg_type"] == "single_photo"
        assert sent[2]["msg_type"] == "single_photo"

    def test__send_multi_img(self, telegram_msgr: TelegramMessenger, tmp_path: Path):
        # Arrange
        img_path_001 = tmp_path / "img001.png"
        img_path_002 = tmp_path / "img002.png"
        img_paths = [img_path_001, img_path_002]
        for img_path in img_paths:
            write_image(img_path)

        # Act
        sent, exc = telegram_msgr.message_users(img_paths=img_paths)

        # Arrange
        assert len(sent) == 3
        assert sent[0]["msg_type"] == "media_group"
        assert sent[1]["msg_type"] == "media_group"
        assert sent[2]["msg_type"] == "media_group"

    def test__skip_missing_img(self, telegram_msgr: TelegramMessenger, tmp_path: Path):
        # Arrange
        img_path_001 = tmp_path / "img001.png"
        img_path_002 = tmp_path / "img002.png"
        img_path_003 = tmp_path / "img003.png"
        img_paths = [img_path_001, img_path_002, img_path_003]
        write_image(img_path_001)
        write_image(img_path_002)

        # Act
        sent, exc = telegram_msgr.message_users(img_paths=img_paths)

        # Arrange
        assert len(sent) == 3
        assert sent[0]["msg_type"] == "media_group"
        assert sent[0]["length"] == 2
        assert sent[1]["msg_type"] == "media_group"
        assert sent[0]["length"] == 2
        assert sent[2]["msg_type"] == "media_group"
        assert sent[0]["length"] == 2

        assert len(exc) == 1
        assert f"Image img003.png missing!" in exc[0]

    def test__send_to_sudoers(self, telegram_msgr: TelegramMessenger, tmp_path: Path):
        # Act
        sent, exc = telegram_msgr.message_users(users="sudoers", texts="some message")

        # Assert
        assert len(sent) == 1
        assert sent[0]["user"] == 901

    def test__exception_caught(self, telegram_msgr: TelegramMessenger):
        # Act
        with pytest.warns(UserWarning):
            # texts as list of non-str will fail!
            sent, exc = telegram_msgr.message_users(texts=[[]])

        # Assert
        assert len(sent) == 0

        assert len(exc) == 3
