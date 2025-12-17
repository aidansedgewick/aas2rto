import os
import shutil
import subprocess
import time
from logging import getLogger
from pathlib import Path

import pandas as pd

from jinja2 import Environment, FileSystemLoader

from astropy.time import Time

from aas2rto import utils
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager
from aas2rto.target import Target

logger = getLogger(__name__.split(".")[-1])


class StaticPagesManager:

    default_config = {
        "publish_interval": 1800.0,
        "git": None,
    }
    REQUIRED_GIT_PARAMETERS = ("user_email", "remote_url", "remote_name")
    EXPECTED_GIT_PARAMETERS = (*REQUIRED_GIT_PARAMETERS, "branch")

    def __init__(
        self, config: dict, outputs_manager: OutputsManager, path_manager: PathManager
    ):
        self.config = self.default_config.copy()
        self.config.update(config)
        utils.check_unexpected_config_keys(
            self.config, self.default_config, name="web.static_pages", raise_exc=True
        )

        self.outputs_manager = outputs_manager
        self.path_manager = path_manager

        self.prepare_web_directories()  # Do this first.

        self.check_git_config()
        self.prepare_git_repo()

        self.web_environment = Environment(
            loader=FileSystemLoader(self.path_manager.web_templates_path)
        )
        self.last_git_publish = time.perf_counter()

    def prepare_web_directories(self):
        self.web_base_path = self.path_manager.web_path / "static"
        self.web_base_path.mkdir(exist_ok=True, parents=True)

        self.index_path = self.web_base_path / "index.html"

        self.web_im_path = self.web_base_path / "im"
        self.web_im_path.mkdir(exist_ok=True, parents=True)

        self.web_lists_path = self.web_base_path / "lists"
        self.web_lists_path.mkdir(exist_ok=True, parents=True)

        self.web_target_path = self.web_base_path / "target"
        self.web_target_path.mkdir(exist_ok=True, parents=True)

    def check_git_config(self):
        self.git_config = self.config.get("git", None)
        if self.git_config is None:
            logger.info("Can't use git; no git config provided.")
            self.git_config = None
            return

        utils.check_unexpected_config_keys(
            self.git_config, self.EXPECTED_GIT_PARAMETERS, raise_exc=True
        )
        utils.check_missing_config_keys(
            self.git_config, self.REQUIRED_GIT_PARAMETERS, raise_exc=True
        )
        if self.git_config.get("branch", None) is None:
            self.git_config["branch"] == "main"

    def prepare_git_repo(self):
        if self.git_config is None:
            logger.info("Will not prepare .git directory")
            return

        remote_name = self.git_config["remote_name"]
        remote_url = self.git_config["remote_url"]

        git_dir = self.web_base_path / ".git"
        if not git_dir.exists():
            try:
                logger.info("first try to clone an existing repo (at depth=1)...")
                self.clone_existing_repo()
            except subprocess.CalledProcessError as e:
                logger.error("no existing repo available")
                self.init_new_git_repo()
        else:
            logger.info(".git already initialised")

        git_remote_cmd = f"git -C {self.web_base_path} remote -v"
        remote_output = subprocess.check_output(git_remote_cmd.split()).decode("utf-8")
        if not remote_output:
            logger.info("set remote origin")
            git_add_origin_cmd = (
                f"git -C {self.web_base_path} remote add {remote_name} {remote_url}"
            )
            add_origin_output = subprocess.check_output(
                git_add_origin_cmd.split()
            ).decode("utf-8")

    def clone_existing_repo(self):
        remote_url = self.git_config["remote_url"]

        git_clone_command = f"git clone --depth 1 {remote_url} {self.web_base_path}"
        output = subprocess.check_output(git_clone_command.split())

        git_dir = self.web_base_path / ".git"
        if git_dir.exists():
            logger.info("cloned - success!")
        else:
            logger.warning("clone exited ok, but no cloned repo...")

    def init_new_git_repo(self):

        user_email = self.git_config["user_email"]
        branch = self.git_config["branch"]

        git_init_cmd = f"git -C {self.web_base_path} init"
        logger.info(f"initialise git repo in {self.web_base_path}:\n    {git_init_cmd}")
        output = subprocess.check_output(git_init_cmd.split())

        git_email_cmd = f"git -C {self.web_base_path} config user.email {user_email}"
        email_output = subprocess.check_output(git_email_cmd.split())

        git_branch_cmd = f"git -C {self.web_base_path} branch -m '{branch}'"
        branch_output = subprocess.check_output(git_branch_cmd.split())

    def _get_sci_ranked_page_path(self):
        return self.web_lists_path / "sci_ranked.html"

    def _get_sci_ranked_list_url(self):
        return self._get_sci_ranked_page_path().relative_to(self.web_base_path)

    def _get_obs_ranked_page_path(self, obs_name: str):
        return self.web_lists_path / f"{obs_name}_ranked.html"

    def _get_obs_ranked_list_url(self, obs_name: str):
        return self._get_obs_ranked_page_path(obs_name).relative_to(self.web_base_path)

    def _get_obs_visible_page_path(self, obs_name: str):
        return self.web_lists_path / f"{obs_name}_visible.html"

    def _get_obs_visible_list_url(self, obs_name: str):
        return self._get_obs_visible_page_path(obs_name).relative_to(self.web_base_path)

    def _get_target_page_path(self, target_id: str):
        return self.web_target_path / f"{target_id}.html"

    def _get_target_page_url(self, target_id: str):
        return self._get_target_page_path(target_id).relative_to(self.web_base_path)

    def build_webpages(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.build_index_page(t_ref=t_ref)

        self.build_science_ranked_page(t_ref=t_ref)
        self.build_obs_ranked_pages(t_ref=t_ref)
        self.build_obs_visible_pages(t_ref=t_ref)

        for target_id, target in self.outputs_manager.target_lookup.items():
            if target.updated:
                self.build_webpage_for_target(target, t_ref=t_ref)
                self.build_redirect_pages_for_target(target)

    def build_index_page(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%H:%M %a, %d %b %Y")

        page_data = dict(
            science_ranked_url=self._get_sci_ranked_list_url(),
            obs_ranked_url_lookup={},
            obs_visible_url_lookup={},
            update_time_str=t_str,
        )
        for obs_name in self.outputs_manager.obs_ranked_lists.keys():
            page_data["obs_ranked_url_lookup"][obs_name] = (
                self._get_obs_ranked_list_url(obs_name)
            )
        for obs_name in self.outputs_manager.obs_ranked_lists.keys():
            page_data["obs_visible_url_lookup"][obs_name] = (
                self._get_obs_visible_list_url(obs_name)
            )

        template = self.web_environment.get_template("index.html")
        rendered_page = template.render(page_data)

        with open(self.index_path, "w+") as f:
            f.writelines(rendered_page)

    def row_data_from_df(self, df: pd.DataFrame):
        row_data = []
        for idx, row in df.iterrows():
            target_id = row["target_id"]
            data = [v for k, v in row.items() if k != "target_id"]

            target_page_url = f"../{self._get_target_page_url(target_id)}"
            row = {
                "idx": idx,
                "target_id": target_id,
                "target_page_url": target_page_url,
                "data": data,
            }
            row_data.append(row)
        return row_data

    def build_science_ranked_page(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%H:%M %a, %d %b %Y")

        df = self.outputs_manager.science_ranked_list
        page_data = dict(
            title="Science ranked",
            page_header=f"Targets ranked by science score",
            column_names=list(df.columns),
            row_data=self.row_data_from_df(df),
            index_url="../index.html",
            update_time_str=t_str,
        )

        template = self.web_environment.get_template("list.html")
        rendered_page = template.render(page_data)

        list_page_path = self._get_sci_ranked_page_path()
        with open(list_page_path, "w+") as f:
            f.writelines(rendered_page)

    def build_obs_ranked_pages(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%H:%M %a, %d %b %Y")

        for obs_name, df in self.outputs_manager.obs_ranked_lists.items():

            page_data = dict(
                title=f"{obs_name} ranked",
                page_header=f"Ranked targets at {obs_name}",
                column_names=list(df.columns),
                row_data=self.row_data_from_df(df),
                index_url="../index.html",
                update_time_str=t_str,
            )

            template = self.web_environment.get_template("list.html")
            rendered_page = template.render(page_data)

            list_page_path = self._get_obs_ranked_page_path(obs_name)
            with open(list_page_path, "w+") as f:
                f.writelines(rendered_page)

    def build_obs_visible_pages(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%H:%M %a, %d %b %Y")

        for obs_name, df in self.outputs_manager.obs_visible_lists.items():
            page_data = dict(
                title=f"{obs_name} visible",
                page_header=f"Visible targets at {obs_name}",
                column_names=list(df.columns),
                row_data=self.row_data_from_df(df),
                index_url="../index.html",
                update_time_str=t_str,
            )
            template = self.web_environment.get_template("list.html")
            rendered_page = template.render(page_data)

            list_page_path = self._get_obs_visible_page_path(obs_name)
            with open(list_page_path, "w+") as f:
                f.writelines(rendered_page)

    def build_webpage_for_target(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%H:%M %a, %d %b %Y")

        if target.lc_fig_path is not None:
            lc_name = target.lc_fig_path.name
            web_lc_fig_path = self.web_im_path / f"{lc_name}"
            if target.lc_fig_path.exists():
                shutil.copy2(target.lc_fig_path, web_lc_fig_path)
            im_path = f"../{web_lc_fig_path.relative_to(self.web_base_path)}"
            rel_lc_path = im_path
        else:
            rel_lc_path = ""

        rel_vis_paths = []
        for obs, vis_path in target.vis_fig_paths.items():
            web_vis_path = self.web_im_path / f"{vis_path.name}"
            if vis_path.exists():
                shutil.copy2(vis_path, web_vis_path)

            im_path = f"../{web_vis_path.relative_to(self.web_base_path)}"
            rel_vis_paths.append(im_path)

        additional_path_list = []
        for fig_name, additional_fig_path in target.additional_fig_paths.items():
            web_fig_path = self.web_im_path / f"{additional_fig_path.name}"
            if additional_fig_path.exists():
                shutil.copy2(additional_fig_path, web_fig_path)
            im_path = f"../{web_fig_path.relative_to(self.web_base_path)}"
            additional_path_list.append()

        target_info_str = target.get_info_string(
            header_open="<h2>", header_close="</h2>"
        )
        target_info_str = target_info_str.replace("\n", "<br>")
        page_data = dict(
            target_id=target.target_id,
            target_text=target_info_str,
            lc_fig_path=rel_lc_path,
            vis_fig_path_list=rel_vis_paths,
            additional_fig_path_list=additional_path_list,
            index_url="../index.html",
            update_time_str=t_str,
        )

        template = self.web_environment.get_template("target.html")
        rendered_page = template.render(page_data)

        target_page_path = self._get_target_page_path(target.target_id)

        with open(target_page_path, "w+") as f:
            f.writelines(rendered_page)

    def build_redirect_pages_for_target(self, target: Target):

        for alt_key, alt_id in target.alt_ids.items():
            if alt_id == target.target_id:
                continue  # DON'T rewrite real page data with redirect - endless loop!
            template = self.web_environment.get_template("target_redirect.html")
            page_data = dict(
                redirect_url=self._get_target_page_url(alt_id),
                target_id=target.target_id,
            )
            rendered_page = template.render(page_data)
            redirect_page_path = self._get_target_page_path(alt_id)
            with open(redirect_page_path, "w+") as f:
                f.writelines(rendered_page)

    def refresh_tmux_ssh_auth_sock(self):
        """
        In tmux, the env is not kept up to date (?) by default,
        so need to reload the $SSH_AUTH_SOCK parameter, otherwise
        git push will fail.
        """

        is_tmux_session = os.environ.get("TMUX", False)

        if not is_tmux_session:
            return

        ssh_auth_sock_path = Path(os.environ["SSH_AUTH_SOCK"])
        if ssh_auth_sock_path.exists():
            logger.info("no need to update $SSH_AUTH_SOCK - still exists!")
            return

        logger.info("refresh $SSH_AUTH_SOCK...")
        tmux_env_cmd = "tmux show-env -s"
        tmux_env_lines = (
            subprocess.check_output(tmux_env_cmd.split()).decode().split("\n")
        )

        for line in tmux_env_lines:
            commands = line.split(";")
            # the relevant line is something like:
            # "SSH_AUTH_SOCK='/tmp/ssh_blah/agent.1234'; export SSH_AUTH_SOCK"
            if commands[0].startswith("SSH_AUTH_SOCK"):
                key, ssh_auth_sock_str = commands[0].split("=")
                ssh_auth_sock = ssh_auth_sock_str.replace('"', "")  # no quotes
                print(f"choose {ssh_auth_sock}")

        os.environ["SSH_AUTH_SOCK"] = ssh_auth_sock
        logger.info(f"choose {ssh_auth_sock}")
        logger.info("new env var:", os.environ["SSH_AUTH_SOCK"])

        ssh_auth_sock_path = Path(ssh_auth_sock)
        if ssh_auth_sock_path.exists():
            logger.info("newly set $SSH_AUTH_SOCK exists!")
        else:
            msg = "something went wrong - new $SSH_AUTH_SOCK does not exist..."
            logger.warning(msg)

    def publish_to_git(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if self.git_config is None:
            logger.info("no git config. skip attempting git commit.")

        git_branch = self.git_config["branch"]

        commit_interval = self.config["publish_interval"]
        elapsed = time.perf_counter() - self.last_git_publish
        if elapsed < commit_interval:
            msg = f"skip git publish: {elapsed:.1f}s<{commit_interval:.1f} required"
            logger.info(msg)
            return
        self.last_git_publish = time.perf_counter()

        git_add_cmd = f"git -C {self.local_www_path} add --all"
        git_add_output = subprocess.check_output(git_add_cmd.split()).decode("utf-8")

        git_commit_cmd = f"git -C {self.local_www_path} commit -m 'Update_{t_ref.isot}' --allow-empty"
        try:
            commit_output = subprocess.check_output(git_commit_cmd.split()).decode(
                "utf-8"
            )
        except subprocess.CalledProcessError as e:
            print(e)

        self.refresh_tmux_ssh_auth_sock()

        git_push_cmd = f"git -C {self.local_www_path} push --set-upstream {self.remote_name} {self.git_branch} --force"
        try:
            push_output = subprocess.check_output(git_push_cmd.split()).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(e)

    def publish(self):
        self.publish_to_git()
        # Other publishers go here...

    def perform_all_tasks(self):

        self.build_webpages()
        self.publish()
