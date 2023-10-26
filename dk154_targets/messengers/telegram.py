import time
import requests
import traceback
import yaml
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from astropy.time import Time

try:
    import telegram
except ModuleNotFoundError as e:
    telegram = None

logger = getLogger(__name__.split(".")[-1])


class TelegramMessenger:
    http_url = "https://api.telegram.org"

    expected_kwargs = ("token", "users", "sudoers", "users_file")

    def __init__(self, telegram_config: dict):
        self.telegram_config = telegram_config
        self.token = self.telegram_config.get("token", None)
        users = self.telegram_config.get("users", {})
        sudoers = self.telegram_config.get("sudoers", {})
        users_file = self.telegram_config.get("users_file", None)

        for key, val in self.telegram_config.keys():
            if key not in self.expected_kwargs:
                logger.warning(f"unexpeted telegram_config kwarg: {key}")

        if isinstance(users, list):
            logger.warning("in telegram config 'users', use a dict to give names!")
            users = {u: "unknown_user" for u in users}
        self.users = users
        if isinstance(users, list):
            logger.warning("in telegram config 'sudoers', use a dict to give names!")
            sudoers = {u: self.users.get(u, "unknown_sudoer") for u in sudoers}
        self.sudoers = sudoers
        self.users.update(self.sudoers)

        if users_file is not None:
            self.users_file = Path(users_file)

        self.bot = None
        if telegram is None:
            msg = (
                "\033[31;1mtelegram module did not import correctly\033[0m.\n"
                "try \033[36;1mpython3 -m pip install python-telegram-bot\033[0m"
            )
            return
        if self.token is None:
            logger.warning("no token provided in telegram config.")
            return

        try:
            self.bot = telegram.Bot(token=self.token)
        except Exception as e:
            logger.warning(f"during telegram bot init:\n{e}")
        return

    def read_new_users(self):
        if self.users_file is not None:
            if self.users_file.exists():
                with open(self.users_file, "r") as f:
                    new_users = yaml.load(f, loader=yaml.FullLoader)
                if isinstance(new_users, list):
                    new_users = {u: "unknown_user" for u in self.users}
                return new_users
        return {}

    def send_to_user(
        self, user, texts: List[str] = None, img_paths: List[str] = None, caption=None
    ):
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
        if img_paths is not None:
            if isinstance(img_paths, str) or isinstance(img_paths, Path):
                img_paths = [img_paths]

        if texts is not None:
            for text in texts:
                # url = f"{self.http_url}/bot{self.token}/sendMessage?chat_id={user}&text={text}"
                # requests.get(url)
                self.bot.send_message(
                    chat_id=user, text=text, disable_web_page_preview=True
                )
        if img_paths is not None:
            img_list = []
            for img_path in img_paths:
                if not img_path.exists():
                    logger.warning(f"{Path(img_path).stem} missing")
                with open(img_path, "rb") as img:
                    if len(img_paths) > 2:
                        img_list.append(telegram.InputMediaPhoto(img))
                    else:
                        self.bot.send_photo(chat_id=user, photo=img, caption=caption)
            if len(img_list) > 2:
                self.bot.send_media_group(chat_id=user, media=img_list)
            return None

    def message_users(
        self,
        users=None,
        texts: List[str] = None,
        img_paths: List[str] = None,
        caption=None,
    ):
        if self.bot is None:
            logger.warning("telegram bot did not initialise")
            return

        if users is None:
            try:
                new_users = self.read_user_file()
            except Exception as e:
                new_users = {}
            users = {**self.users, **new_users}
        if users in ["sudoers", "sudo"]:
            users = self.sudoers
        exceptions = []
        for user, user_name in users:
            try:
                self.send_to_user(
                    user, texts=texts, img_paths=img_paths, caption=caption
                )
            except Exception as e:
                tr = traceback.format_exc()
                exceptions.append(f"for user {user} ({user_name}):\n{e}\n\n")
        if len(exceptions) > 0:
            execption_str = "\n\nand\n\n".join(e for e in exceptions)
            msg = "During message_users:\n\n" + execption_str
            for sudoer in self.sudoers:
                try:
                    self.send_to_user(sudoer, texts=msg)
                except Exception as e:
                    tr = traceback.format_exc()
                    logger.warn(
                        f"Exception while sending error report to telegram sudoers! {e}\n{tr}"
                    )
        return None

    # def send_crash_report(self, info: str = None, t_ref: Time = None):
    #    tr = traceback.format_exc()
    #    self.message_users(users="sudoers", texts=[info, tr])
