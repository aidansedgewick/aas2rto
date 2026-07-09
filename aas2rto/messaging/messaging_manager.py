import time
import traceback
from collections import defaultdict
from logging import getLogger
from typing import Callable

from astropy.time import Time

from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup

from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.messaging.slack_messenger import SlackMessenger
from aas2rto.target import Target

# from aas2rto.messaging.html_webpage_manager import HtmlWebpageManager
from aas2rto.utils import format_link_as_html, format_link_as_markdown

logger = getLogger(__name__.split(".")[-1])

EXPECTED_MESSENGERS = {
    "slack": SlackMessenger,
    "telegram": TelegramMessenger,
}


class DefaultMessageFilter:

    def __init__(self, minimum_score: float = 0.0, message_interval_hrs: float = 3.0):
        self.minimum_score = minimum_score
        self.message_interval_hrs = message_interval_hrs
        self.logger = getLogger("message_filter")

    def __call__(self, target: Target, t_ref: Time = None):

        if not target.updated:
            self.logger.debug(f"{target.target_id} not updated; skip")
            return False, "not updated"

        if len(target.info_messages) == 0:
            return False, "no messages"

        last_score = target.get_latest_science_score()
        if last_score is None:
            self.logger.debug(f"{target.target_id} last score is None; skip")
            return False, "no score"

        if last_score < self.minimum_score:
            dbg = f"{target.target_id} last score: {last_score} < {self.minimum_score}; skip"
            self.logger.debug(dbg)
            return False, "low score"

        if target.last_message_time is not None:
            message_dt = t_ref - target.last_message_time
            if message_dt.to("hour").value < self.message_interval_hrs:
                return False, "recent message"

        return True, "passed"


class MessagingManager:

    def __init__(
        self,
        messengers_config: dict[str, dict],
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):
        self.messengers_config = messengers_config

        self.target_lookup = target_lookup
        self.path_manager = path_manager

        self.initialize_messengers()

        self.default_message_filter = DefaultMessageFilter()

    def initialize_messengers(self):
        self.messengers = {}
        self.telegram_messenger = None
        self.slack_messenger = None

        for msgr_name, msgr_config in self.messengers_config.items():

            use_msgr = msgr_config.pop("use", True)
            if not use_msgr:
                logger.info(f"Skip messenger {msgr_name} init")
                continue
            if msgr_name == "telegram":
                msgr = TelegramMessenger(msgr_config)
                self.telegram_messenger = msgr
            elif msgr_name == "slack":
                msgr = SlackMessenger(msgr_config)
                self.slack_messenger = msgr
            else:
                raise NotImplementedError(f"No messenger {msgr_name}")
            self.messengers[msgr_name] = msgr

    def send_sudo_messages(self, texts):
        if self.telegram_messenger is not None:
            sent, exc = self.telegram_messenger.message_users(
                texts=texts, users="sudoers"
            )
            return sent, exc
        return [], []

    def perform_messaging_tasks(
        self, message_filter: Callable = None, t_ref: Time = None
    ):
        """
        Loop through each target in TargetLookup.
        If there are any messages attached to a target, send them via the available messengers.
        Messages are (mostly) attached to targets in the TargetLookup.

        Parameters
        ----------
        t_ref
        """

        t_ref = t_ref or Time.now()

        message_filter = message_filter or self.default_message_filter

        skipped = []
        no_updates = []
        skipped_reasons = defaultdict(list)
        sent = []

        logger.info("perform messaging tasks")
        for target_id, target in self.target_lookup.items():
            logger.debug(f"messaging for {target_id}")

            if len(target.info_messages) == 0:
                no_updates.append(target.target_id)

            allow_message = message_filter(target, t_ref=t_ref)
            if isinstance(allow_message, tuple):
                allow_message, reason = allow_message
            else:
                reason = "<unknown-reason>"
            if not allow_message:
                skipped.append(target.target_id)
                skipped_reasons[reason].append(target.target_id)
                continue

            lc_fig_path = target.lc_fig_path
            # vis_fig_paths = []
            # for obs_name, vis_fig_path in target.vis_fig_paths.items():
            #     vis_fig_paths.append(vis_fig_path)

            if self.telegram_messenger is not None:
                # NO formatter for telegram messages.
                messages = target.get_info_lines() + target.info_messages
                message_text = "\n".join(msg for msg in messages)
                try:
                    self.telegram_messenger.message_users(texts=message_text)
                    self.telegram_messenger.message_users(img_paths=lc_fig_path)
                    # self.telegram_messenger.message_users(
                    #    texts="visibilities", img_paths=vis_fig_paths
                    # )
                except Exception as e:
                    logger.error(f"error in telegram messaging {e}")

            if self.slack_messenger is not None:
                messages = (
                    target.get_info_lines(link_formatter=format_link_as_markdown)
                    + target.info_messages
                )
                try:
                    self.slack_messenger.send_messages(
                        texts=message_text, img_paths=lc_fig_path
                    )
                except Exception as e:
                    logger.error(f"error in slack messaging, {e}")
            sent.append(target_id)
            time.sleep(0.5)

        logger.info(f"skipped messages for {len(skipped)} targets")
        reason_str = "reasons:\n" + "\n".join(
            f"    {r}: {len(rs)}" for r, rs in skipped_reasons.items()
        )
        logger.info(reason_str)

        logger.info(f"sent messages for {len(sent)} targets")
        return sent, skipped, no_updates

    def send_crash_reports(self, text: str = None):
        """
        Convenience method for sending most recent traceback via telegram to sudoers.

        Parameters
        ----------
        text : str or list of str, default=None
            str or list of str of text to accompany the traceback
        """
        if text is None:
            text = []
        if isinstance(text, str):
            text = [text]
        tr = text + [traceback.format_exc()]
        logger.error("\n" + "\n".join(tr))
        if self.telegram_messenger is not None:
            sent, exc = self.telegram_messenger.message_users(users="sudoers", texts=tr)
            return sent, exc
        return [], []
