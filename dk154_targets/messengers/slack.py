import os
from logging import getLogger
from pathlib import Path

try:
    import slack_sdk
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except Exception as e:
    slack_sdk = None

from dk154_targets import paths

logger = getLogger(__name__.split(".")[-1])


class SlackMessenger:
    def __init__(self, slack_config):
        self.slack_config = slack_config
        self.token = self.slack_config.get("token", None)
        self.channel_id = self.slack_config.get("channel_id", None)

        self.client = None  # stays as None if can't init.
        if slack_sdk is None:
            logger.warning(f"\n{e}")
            logger.warning("\033[31;1merror importing slack_sdk\033[0m")

        if self.token is None:
            logger.warning("no 'token' in config: can't init client.")
            return

        if self.channel_id is None:
            logger.warning("no 'channel_id' in config.")
            return

        try:
            self.client = WebClient(token=self.token)
        except Exception as e:
            logger.warning(f"during slack client init:\n{e}")

        return

    def send_messages(self, texts, img_paths):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(img_paths, str) or isinstance(img_paths, Path):
            img_paths = [img_paths]

        if self.client is None:
            logger.warning("slack client did not initialise.")

        for text in texts:
            try:
                self.client.chat_postMessage(channel=self.channel_id, text=text)
            except SlackApiError as e:
                logger.warning(e)

        for img_path in img_paths:
            if img_path is None:
                continue
            with open(img_path, "rb") as img:
                try:
                    result = self.client.files_upload_v2(
                        channel=self.channel_id,
                        initial_comment="file here :rocket:",
                        file=img,  # Path(img_path),
                    )
                except SlackApiError as e:
                    logger.warning(e)
        return
