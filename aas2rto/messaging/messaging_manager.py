import time
import traceback
from collections import defaultdict
from logging import getLogger
from typing import Callable

from astropy.time import Time

from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto.utils import format_link_as_html, format_link_as_markdown

from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.messaging.slack_messenger import SlackMessenger

# from aas2rto.messaging.html_webpage_manager import HtmlWebpageManager

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

    def __call__(self, target: Target, t_ref: Time = None) -> tuple[bool, list[str]]:
        t_ref = t_ref or Time.now()

        allow = True
        reasons = []
        if not target.updated:
            self.logger.debug(f"{target.target_id} not updated; skip")
            allow = False
            reasons.append("not updated")

        if len(target.info_messages) == 0:
            allow = False
            reasons.append("no messages")

        last_score = target.get_latest_science_score()
        if last_score is None:
            self.logger.debug(f"{target.target_id} last score is None; skip")
            allow = False
            reasons.append("no science score")
        else:
            if last_score < self.minimum_score:
                dbg = (
                    f"{target.target_id} last score: "
                    f"{last_score} < {self.minimum_score}; skip"
                )
                self.logger.debug(dbg)
                allow = False
                reasons.append("low score")

        if isinstance(target.last_message_time, Time):
            message_dt = t_ref - target.last_message_time
            if message_dt.to("hour").value < self.message_interval_hrs:
                reasons.append("recent message")
                allow = False

        return allow, reasons


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

        message_filter = message_filter or DefaultMessageFilter()

        result = {"sent": [], "skipped": [], "error": []}
        result_reasons = defaultdict(list)

        logger.info("perform messaging tasks")
        for target_id, target in self.target_lookup.items():
            logger.debug(f"messaging for {target_id}")

            allow_message, allow_reasons = message_filter(target, t_ref=t_ref)
            if isinstance(allow_reasons, str):
                allow_reasons = [allow_reasons]  # should be a list.
            if not allow_message:
                result["skipped"].append(target.target_id)
                for reason in allow_reasons:
                    result_reasons[reason].append(target.target_id)
                continue

            lc_fig_path = target.lc_fig_path
            # vis_fig_paths = []
            # for obs_name, vis_fig_path in target.vis_fig_paths.items():
            #     vis_fig_paths.append(vis_fig_path)

            sending_failed = False
            if self.telegram_messenger is not None:

                messages = target.get_info_lines(link_formatter=format_link_as_html)
                messages.extend(target.info_messages)
                message_text = "\n".join(msg for msg in messages)
                try:
                    self.telegram_messenger.message_users(texts=message_text)
                    self.telegram_messenger.message_users(img_paths=lc_fig_path)
                    # self.telegram_messenger.message_users(
                    #    texts="visibilities", img_paths=vis_fig_paths
                    # )
                except Exception as e:
                    result["error"].append(target.target_id)
                    result_reasons["slack_failed"].append(target.target_id)
                    logger.error(f"error in telegram messaging {e}")
                    sending_failed = True

            if self.slack_messenger is not None:
                messages = target.get_info_lines(link_formatter=format_link_as_markdown)
                messages.extend(target.info_messages)
                message_text = "\n".join(msg for msg in messages)
                try:
                    self.slack_messenger.send_messages(
                        texts=message_text, img_paths=lc_fig_path
                    )
                except Exception as e:
                    result["error"].append(target.target_id)
                    result_reasons["telegram_failed"].append(target.target_id)
                    logger.error(f"error in slack messaging, {e}")
                    sending_failed = True

            if not sending_failed:
                target.last_message_time = t_ref

            result["sent"].append(target_id)
            time.sleep(0.3)

        logger.info(f"sent messages for {len(result['sent'])}")
        logger.info(f"skipped messages for {len(result['skipped'])}")
        skipped_str = "reasons (and/or):\n" + "\n".join(
            f"    {r_str}: {len(ids)}" for r_str, ids in result_reasons.items()
        )
        logger.info(skipped_str)

        return result, dict(result_reasons)  # dict stops 'default' behaviour.

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
