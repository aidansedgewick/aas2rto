import time
import traceback
from logging import getLogger

from astropy.time import Time


from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup


from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.messaging.slack_messenger import SlackMessenger
from aas2rto.messaging.git_webpage_manager import GitWebpageManager

logger = getLogger(__name__.split(".")[-1])


class MessagingManager:

    def __init__(
        self,
        messengers_config: dict,
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
        self.git_webpage_manager = None
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
            elif msgr_name == "git_web":
                msgr = GitWebpageManager(
                    msgr_config, self.path_manager.lookup.get("www_path", None)
                )
                self.git_webpage_manager = msgr
            else:
                raise NotImplementedError(f"No messenger {msgr_name}")
            self.messengers[msgr_name] = msgr

    def send_sudo_messages(self, texts):
        if self.telegram_messenger is not None:
            self.telegram_messenger.message_users(texts=texts, users="sudoers")

    def perform_messaging_tasks(self, t_ref: Time = None):
        """
        Loop through each target in TargetLookup.
        If there are any messages attached to a target, send them via the available messengers.
        Messages are (mostly) attached to targets in the TargetLookup.

        Parameters
        ----------
        t_ref
        """

        t_ref = t_ref or Time.now()

        skipped = []
        no_updates = []
        sent = []

        minimum_score = self.selector_parameters.get("minimum_score")
        logger.info("perform messaging tasks")
        for target_id, target in self.target_lookup.items():
            logger.debug(f"messaging for {target_id}")
            if len(target.update_messages) == 0:
                logger.debug(f"no messages")
                no_updates.append(target_id)
                continue

            if not target.updated:
                logger.debug(f"not updated; skip")
                skipped.append(target_id)
                continue

            last_score = target.get_last_score()
            if last_score is None:
                skipped.append(target_id)
                logger.debug(f"last score is None; skip")
                continue

            if last_score < minimum_score:
                skipped.append(target_id)
                logger.debug(f"last score: {last_score} < {minimum_score}; skip")
                continue

            intro = f"Updates for {target_id}"
            messages = [target.get_info_string()] + target.update_messages
            message_text = "\n".join(msg for msg in messages)

            lc_fig_path = target.lc_fig_path
            vis_fig_paths = []
            for obs_name, vis_fig_path in target.vis_fig_paths.items():
                vis_fig_paths.append(vis_fig_path)

            if self.telegram_messenger is not None:
                self.telegram_messenger.message_users(texts=message_text)
                self.telegram_messenger.message_users(img_paths=lc_fig_path)
                self.telegram_messenger.message_users(
                    texts="visibilities", img_paths=vis_fig_paths
                )
            if self.slack_messenger is not None:
                self.slack_messenger.send_messages(
                    texts=message_text, img_paths=lc_fig_path
                )
            sent.append(target_id)
            time.sleep(0.5)

        logger.info(f"no updates to send for {len(no_updates)} targets")
        logger.info(f"skipped messages for {len(skipped)} targets")
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
            self.telegram_messenger.message_users(users="sudoers", texts=tr)
