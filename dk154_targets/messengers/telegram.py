import time
import requests
import traceback
from logging import getLogger
from pathlib import Path
from typing import Dict, List

try:
    import telegram
except ModuleNotFoundError as e:
    telegram = None

logger = getLogger(__name__.split(".")[-1])


class TelegramMessenger:
    http_url = "https://api.telegram.org"

    def __init__(self, telegram_config: dict):
        self.telegram_config = telegram_config
        self.token = self.telegram_config.get("token", None)
        self.users = self.telegram_config.get("users", [])
        self.sudoers = self.telegram_config.get("sudoers", [])
        self.users_file = self.telegram_config.get("users_file", None)

        for sudoer in self.sudoers:
            if sudoer not in self.users:
                self.users.append(sudoer)

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
            with open(self.users_file, "r") as f:
                lines = f.readlines()
            return [l.split("#").strip() for l in lines]
        return []

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
                self.bot.send_message(chat_id=user, text=text)
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
                new_users = []
            users = list(set(self.users + new_users))
        elif users in ["sudoers", "sudo"]:
            users = self.sudoers
        exceptions = []
        for user in self.users:
            try:
                self.send_to_user(
                    user, texts=texts, img_paths=img_paths, caption=caption
                )
            except Exception as e:
                tr = traceback.format_exc()
                exceptions.append(f"for user {user}:\n{e}\n\n{tr}")
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

    def send_crash_report(self, texts: List[str] = None):
        t_ref = t_ref or Time.now()

        t_str = t_ref.strftime("%Y%m%d %H:%M:%S")
        crash_reports = [f"CRASH at UT {t_str}"] + texts
        self.message_users(users="sudoers", texts=crash_reports)
