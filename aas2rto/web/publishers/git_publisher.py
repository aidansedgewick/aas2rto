from __future__ import annotations

import os
import shlex
import subprocess
import time
from logging import getLogger
from pathlib import Path

from astropy.time import Time

from aas2rto.utils import check_unexpected_config_keys, check_missing_config_keys

logger = getLogger("git_publisher")


class GitPublisher:

    default_deploy_key_path = "~/.ssh/aas2rto_deploy_key"
    required_git_parameters = (
        "user_email",
        "remote_url",
        "remote_name",
        "deploy_key_path",
    )
    expected_git_parameters = (*required_git_parameters, "branch")

    def __init__(self, config: dict, web_base_path: Path):

        self.config = config

        if self.config.get("branch", None) is None:
            self.config["branch"] == "main"

        deploy_key_path = self.config.get("deploy_key_path", None)
        if deploy_key_path is None:
            deploy_key_path = self.default_deploy_key_path
        self.config["deploy_key_path"] = Path.expanduser(deploy_key_path)
        if not self.config["deploy_key_path"].exists():
            msg = (
                f"No deploy key found at {deploy_key_path}.\n"
                f"Generate one (no passphrase) with:\n"
                f"  \033[35;1mssh-keygen -t ed25519 -f {deploy_key_path} -N ''\033[0m\n"
                f"then add {deploy_key_path}.pub as a Deploy Key (with write access) "
                f"on your GitHub repo's Settings -> Deploy keys."
            )
            raise FileNotFoundError(msg)

        check_unexpected_config_keys(
            self.config,
            self.expected_git_parameters,
            name="web.static_pages.git",
            raise_exc=True,
        )
        check_missing_config_keys(
            self.config, self.required_git_parameters, raise_exc=True
        )

        self.web_base_path = web_base_path
        self.git_repo_status = None  # Mainly for testing
        self.prepare_git_repo()

    def prepare_git_repo(self):

        git_branch = self.config["branch"]

        git_dir = self.web_base_path / ".git"
        if not git_dir.exists():
            try:
                logger.info("first try to clone an existing repo (at depth=1)...")
                self.clone_existing_repo()
            except subprocess.CalledProcessError as e:
                logger.error("no existing repo available")
                self.init_new_git_repo()
        else:
            logger.info("'.git' already initialised - use existing")
            self.git_repo_status = "existing"  # 'cloned' or 'new_init' set above...

        # Check if remote is known - if not add remote
        git_remote_cmd = f"git -C {self.web_base_path} remote -v"
        cmd = shlex.split(git_remote_cmd)
        remote_output = subprocess.check_output(cmd).decode("utf-8")
        if not remote_output:
            self._git_add_remote()

        # git_branch_command = f"git -C {self.web_base_path} -M {git_branch}"

    def clone_existing_repo(self):
        remote_url = self.config["remote_url"]

        git_clone_command = f"git clone --depth 1 {remote_url} {self.web_base_path}"
        output = subprocess.check_output(shlex.split(git_clone_command))

        git_dir = self.web_base_path / ".git"
        if git_dir.exists():
            logger.info("cloned - success!")
        else:
            logger.warning("clone exited ok, but no cloned repo...")
        self.git_repo_status = "cloned"

    def init_new_git_repo(self):

        user_email = self.config["user_email"]
        branch = self.config["branch"]

        git_init_cmd = f"git -C {self.web_base_path} init"
        logger.info(f"initialise git repo in {self.web_base_path}:\n    {git_init_cmd}")
        output = subprocess.check_output(shlex.split(git_init_cmd))

        git_email_cmd = f"git -C {self.web_base_path} config user.email {user_email}"
        email_output = subprocess.check_output(shlex.split(git_email_cmd))

        git_branch_cmd = f"git -C {self.web_base_path} branch -m '{branch}'"
        branch_output = subprocess.check_output(shlex.split(git_branch_cmd))
        self.git_repo_status = "new_init"

    def _git_add_remote(self):
        remote_name = self.config["remote_name"]
        remote_url = self.config["remote_url"]

        logger.info("set remote origin")
        git_add_remote_cmd = (
            f"git -C {self.web_base_path} remote add {remote_name} {remote_url}"
        )
        cmd = shlex.split(git_add_remote_cmd)
        return subprocess.check_output(cmd).decode("utf-8")

    def publish_to_git(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        commit_interval = self.config["publish_interval"]
        elapsed = time.perf_counter() - self.last_git_publish
        if elapsed < commit_interval:
            msg = f"skip git publish: {elapsed:.1f}s<{commit_interval:.1f} required"
            logger.info(msg)
            return

        self._git_add_all()
        self._git_commit()
        self._git_push()

        self.last_git_publish = time.perf_counter()

    def _git_add_all(self):
        git_add_cmd = f"git -C {self.web_base_path} add --all"
        cmd = shlex.split(git_add_cmd)
        git_add_output = subprocess.check_output(cmd).decode("utf-8")

    def _git_commit(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        # git commit: easier to use commit msg with no space and use easy "split()"
        git_commit_cmd = (
            f"git -C {self.web_base_path} commit -m 'Update_{t_ref.isot}' --allow-empty"
        )
        try:
            cmd = shlex.split(git_commit_cmd)
            commit_output = subprocess.check_output(cmd).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(e)

    def _git_push(self):
        git_branch = self.config["branch"]
        remote_name = self.config["remote_name"]
        deploy_key_path = self.config["deploy_key_filepath"]

        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = f"ssh -i {deploy_key_path} -o IdentitiesOnly=yes"
        env["GIT_TERMINAL_PROMPT"] = "0"  # fail fast

        git_push_cmd = f"git -C {self.web_base_path} push --set-upstream {remote_name} {git_branch} --force"
        try:
            cmd = shlex.split(git_push_cmd)
            push_output = subprocess.check_output(cmd).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(e)
