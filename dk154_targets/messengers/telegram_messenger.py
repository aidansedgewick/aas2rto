import time

# import requests
import traceback
import warnings
import yaml
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from astropy.time import Time

from dk154_targets import utils
from dk154_targets.exc import MissingMediaWarning

try:
    import telegram
except ModuleNotFoundError as e:
    telegram = None

logger = getLogger(__name__.split(".")[-1])


class TelegramMessenger:
    CHUNK_MAX = 4000  # Telegram send fails if text has more than 4096 chars.

    http_url = "https://api.telegram.org"

    expected_kwargs = ("token", "users", "sudoers", "users_file")

    def __init__(self, telegram_config: dict):
        self.telegram_config = telegram_config
        utils.check_unexpected_config_keys(
            self.telegram_config, self.expected_kwargs, name="telegram_config"
        )

        self.token = self.telegram_config.get("token", None)
        if self.token is None:
            logger.warning("\033[33;1mno token provided in telegram_config\033[0m.")

        users = self.telegram_config.get("users", {})
        sudoers = self.telegram_config.get("sudoers", {})
        users_file = self.telegram_config.get("users_file", None)

        if isinstance(users, int):
            users = [users]

        if isinstance(users, list):
            msg = (
                "in telegram config 'users', use a dict to give names!"
                "\n    eg. {01234567: 'user_001', 98765432: 'user_002'}"
            )
            logger.warning(msg)
            users = {u: "unknown_user" for u in users}
        self.users = users

        if isinstance(sudoers, list):
            sudoers = {u: self.users.get(u, "unknown_sudoer") for u in sudoers}
            unknown_sudoers = [
                uid for uid, name in sudoers.items() if name == "unknown_sudoer"
            ]
            if len(unknown_sudoers) > 0:
                msg = (
                    f"unknown_sudoers: {unknown_sudoers}"
                    "in telegram config 'sudoers', use a dict to give names!"
                    f"\n    eg. 01234567: 'sudoer_name'"
                )
                logger.warning(msg)
        self.sudoers = sudoers
        self.users.update(self.sudoers)

        if users_file is not None:
            self.users_file = Path(users_file)
        else:
            self.users_file = None

        self.bot = None
        if telegram is None:
            msg = (
                "\033[31;1mtelegram module did not import correctly\033[0m.\n"
                "try \033[36;1mpython3 -m pip install python-telegram-bot\033[0m"
            )
            logger.warning(msg)
            return

        try:
            self.bot = telegram.Bot(token=self.token)
        except Exception as e:
            logger.error(f"\033[31;1mduring telegram bot init\033[0m]:\n{e}")
        return

    def read_new_users(self):
        if self.users_file is not None:
            if self.users_file.exists():
                with open(self.users_file, "r") as f:
                    new_users = yaml.load(f, Loader=yaml.FullLoader)
                if isinstance(new_users, list):
                    new_users = {u: "unknown_user" for u in new_users}
                return new_users
        return {}

    def send_to_user(
        self,
        user,
        texts: List[str] = None,
        img_paths: List[str] = None,
        caption=None,
        min_media_group_size=2,
    ):

        img_paths = img_paths or []
        if isinstance(img_paths, str) or isinstance(img_paths, Path):
            img_paths = [Path(img_paths)]

        texts = texts or []
        if isinstance(texts, str):
            texts = [texts]

        formatted_texts = []
        for text in texts:
            if len(text) > self.CHUNK_MAX:
                # Telegram send fails if text has more than 4096 chars.
                # formatted_texts.append(f"MESSAGE SPLIT! longer than {self.CHUNK_MAX}")
                text_split = [
                    text[ii : ii + self.CHUNK_MAX]
                    for ii in range(0, len(text), self.CHUNK_MAX)
                ]
                formatted_texts.extend(text_split)
            else:
                formatted_texts.append(text)
        texts = formatted_texts

        sent_messages = []
        for text in texts:
            # url = f"{self.http_url}/bot{self.token}/sendMessage?chat_id={user}&text={text}"
            # requests.get(url)
            sent = self.bot.send_message(
                chat_id=user, text=text, disable_web_page_preview=True
            )
            sent_messages.append(sent)

        # Some explainers...
        # Cannot use send_media_group() for list of imgs which has len==1
        # Cannot use open(img, "rb") [which has type bytes] in send_media_group()
        # Cannot use telegram.InputMediaPhoto in send_photo()
        if img_paths is not None:
            img_list = []
            filenames = []
            # Sort the list of images to send.
            for img_path in img_paths:
                if not Path(img_path).exists():
                    msg = f"{Path(img_path).stem} missing"
                    logger.warning(msg)
                    warnings.warn(MissingMediaWarning(msg))
                    continue
                with open(img_path, "rb") as f:
                    img = f.read()
                img_list.append(img)
                filenames.append(str(img_path))
            # Do the actual sending.
            if len(img_list) >= min_media_group_size:  # probably 2...
                media_list = [
                    telegram.InputMediaPhoto(img, filename=fname)
                    for img, fname in zip(img_list, filenames)
                ]
                sent = self.bot.send_media_group(chat_id=user, media=media_list)
                sent_messages.extend(sent)
                if caption is not None:
                    msg = "will not send caption with send_media_group() (only for single image)"
                    logger.info(msg)
            else:
                for img in img_list:
                    sent = self.bot.send_photo(chat_id=user, photo=img, caption=caption)
                    sent_messages.append(sent)

        return sent_messages

    def message_users(
        self,
        users=None,
        texts: List[str] = None,
        img_paths: List[str] = None,
        caption=None,
    ):
        if self.bot is None:
            logger.warning("telegram bot did not initialise - can't send messages")
            return None, None

        if isinstance(users, int):
            users = [users]

        if isinstance(users, list):
            users = {u: "unknown_user" for u in users}

        if users is None:
            try:
                new_users = self.read_new_users()
            except Exception as e:
                new_users = {}
            users = {**self.users, **new_users}
        if users in ["sudoers", "sudo"]:
            users = self.sudoers

        if isinstance(img_paths, str):
            img_paths = Path(img_paths)

        if isinstance(img_paths, Path):
            img_paths = [img_paths]

        exceptions = []

        img_paths_to_send = []
        if img_paths is not None:
            for img_path in img_paths:
                if not img_path.exists():
                    msg = f"Image {img_path.name} missing!"
                    logger.warning(msg)
                    exceptions.append(msg)
                else:
                    img_paths_to_send.append(img_path)

        sent_messages = []
        for user, user_label in users.items():
            try:
                sent = self.send_to_user(
                    user, texts=texts, img_paths=img_paths_to_send, caption=caption
                )
                sent_messages.extend(sent)
            except Exception as e:
                tr = traceback.format_exc()
                msg = f"for user {user} ({user_label}):\n{tr}\n\n{e}\n\n"
                warnings.warn(UserWarning(msg))
                exceptions.append(msg)

        if len(exceptions) > 0:
            execption_str = "\nand\n".join(e for e in exceptions)
            err_msg = "During message_users:\n\n" + execption_str
            for sudoer, user_label in self.sudoers.items():
                try:
                    self.send_to_user(sudoer, texts=err_msg)
                except Exception as e:
                    tr = traceback.format_exc()
                    logger.warn(
                        f"Exception while sending error report to telegram sudoer"
                        f" {sudoer} ({user_label})! {e}\n{tr}"
                    )
        return sent_messages, exceptions

    # def send_crash_report(self, info: str = None, t_ref: Time = None):
    #    tr = traceback.format_exc()
    #    self.message_users(users="sudoers", texts=[info, tr])
