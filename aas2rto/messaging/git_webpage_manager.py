import os
import shutil
import subprocess
from logging import getLogger
from pathlib import Path

from astropy.time import Time

from aas2rto.target import Target

logger = getLogger(__name__.split(".")[-1])


class GitWebpageManager:

    def __init__(self, www_config, local_www_path):
        self.www_config = www_config

        self.remote_url = self.www_config.get("remote_url", None)
        if self.remote_url is None:
            raise ValueError("must provide 'remote_url' in git_www config.")

        self.git_useremail = self.www_config.get("git_useremail", None)
        if self.git_useremail is None:
            raise ValueError("must privide 'git_useremail'")

        self.remote_name = self.www_config.get("remote_name", None)
        if self.remote_name is None:
            logger.info("set remote name as origin")
            self.remote_name = "origin"

        self.git_branch = self.www_config.get("git_branch", None)
        if self.git_branch is None:
            logger.info("no git branch provided for www. Choose 'main'")
            self.git_branch = "main"

        if local_www_path is None:
            msg = (
                f"In config, if `git_web: use: True`, must provide "
                f"Must provide path for 'www_path' in config_file/paths !"
            )
            raise ValueError(msg)

        self.local_www_path = Path(local_www_path).absolute()
        self.prepare_git_repo()

        self.index_path = self.local_www_path / "index.html"

        self.im_path = self.local_www_path / "im"
        self.im_path.mkdir(exist_ok=True, parents=True)
        self.list_pages_path = self.local_www_path / "list"
        self.list_pages_path.mkdir(exist_ok=True, parents=True)
        self.target_pages_path = self.local_www_path / "target"
        self.target_pages_path.mkdir(exist_ok=True, parents=True)

    def prepare_git_repo(self):

        git_dir = self.local_www_path / ".git"
        if not git_dir.exists():
            try:
                logger.info("first try to clone an existing repo (at depth=1)...")
                self.clone_existing_repo()
            except subprocess.CalledProcessError as e:
                logger.error("existing repo does not exist")
                self.init_new_git_repo()
        else:
            logger.info(".git already initialised")

        git_remote_cmd = f"git -C {self.local_www_path} remote -v"
        remote_output = subprocess.check_output(git_remote_cmd.split()).decode("utf-8")
        if not remote_output:
            logger.info("set remote origin")
            git_add_origin_cmd = f"git -C {self.local_www_path} remote add {self.remote_name} {self.remote_url}"
            add_origin_output = subprocess.check_output(
                git_add_origin_cmd.split()
            ).decode("utf-8")

    def clone_existing_repo(self):
        git_clone_command = (
            f"git clone --depth 1 {self.remote_url} {self.local_www_path}"
        )
        output = subprocess.check_output(git_clone_command.split())

        if self.local_www_path.exists():
            logger.info("cloned - success!")
        else:
            logger.warning("clone exited ok, but no cloned repo...")

    def init_new_git_repo(self):

        git_init_cmd = f"git -C {self.local_www_path} init"
        logger.info(
            f"initialise git repo in {self.local_www_path}:\n    {git_init_cmd}"
        )
        output = subprocess.check_output(git_init_cmd.split())

        git_email_cmd = (
            f"git -C {self.local_www_path} config user.email {self.git_useremail}"
        )
        email_output = subprocess.check_output(git_email_cmd.split())

        git_branch_cmd = f"git -C {self.local_www_path} branch -m '{self.git_branch}'"
        branch_output = subprocess.check_output(git_branch_cmd.split())

    def update_webpages(self, target_lookup, ranked_lists, t_ref=None):
        t_ref = t_ref or Time.now()

        self.build_target_pages(target_lookup, t_ref=t_ref)
        self.build_main_index_page(ranked_lists, t_ref=t_ref)
        self.build_list_index_pages(ranked_lists, t_ref=t_ref)
        self.publish(t_ref=t_ref)

    def _get_list_index_page_path(self, rl_name):
        return self.list_pages_path / f"{rl_name}.html"

    def _get_list_index_page_url(self, rl_name):
        rl_path = self._get_list_index_page_path(rl_name)
        return rl_path.relative_to(self.local_www_path)

    def _get_target_page_path(self, target_id):
        return self.target_pages_path / f"{target_id}.html"

    def _get_target_page_url(self, target_id):
        target_page_path = self._get_target_page_path(target_id)
        return target_page_path.relative_to(self.local_www_path)

    def build_main_index_page(self, ranked_lists, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%a, %d-%b-%Y, %H:%M")

        project_name = self.www_config.get("project_name", "NO PROJECT NAME")

        head_lines = ["<head>", f"<title>{project_name}</title>", f"<head>"]

        header_lines = ["<header>", f"<h>{project_name}<h1>", "</header>"]

        rl_lines = []
        for rl_name in ranked_lists:
            rl_list_url = self._get_list_index_page_url(rl_name)
            rl_lines.append(f"<p><a href={rl_list_url}>{rl_name}</a></p>")

        body_lines = [
            "<body>",
            "<h2>Ranked target lists</h2>",
            f"<p>Last updated {t_str}</p>",
            *rl_lines,
            "</body>",
        ]

        footer_lines = [
            "<footer>",
            f"<p><small>Page updated {t_str}</small></p>",
            "</footer>",
        ]

        page_lines = [
            "<html>",
            *head_lines,
            *header_lines,
            *body_lines,
            *footer_lines,
            "</html>",
        ]

        with open(self.index_path, "w+") as f:
            for line in page_lines:
                f.write(line + "\n")

    def build_list_index_pages(self, ranked_lists, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%a, %d-%b-%Y, %H:%M")

        for rl_name, ranked_list in ranked_lists.items():
            header_lines = [f"<header><h1>targets for {rl_name}</header>"]

            table_lines = ["<table>"]
            header_row = [
                "<tr>",
                "<td>rank</td> <td>name</td> <td>RA</td><td>Dec.</td>",
                "</tr>",
            ]
            table_lines.extend(header_row)

            for ii, row in ranked_list.iterrows():
                target_id = row["target_id"]
                target_page_path = self._get_target_page_path(target_id)
                table_row = [
                    "<tr>",
                    f"<td>{ii+1}</td>",
                    f"<td><a href='../target/{target_page_path.name}'>{target_id}</a></td>",
                    f"<td>{row['ra']:.5f}",
                    f"<td>{row['dec']:.4f}",
                    f"</tr>",
                ]
                table_lines.extend(table_row)
            table_lines.append(f"</table>")

            body_lines = ["<body>", "<p>Targets:</p>", *table_lines, "</body>"]

            footer_lines = [
                "<footer>",
                f"<p><small>Page updated {t_str}</small></p>",
                "</footer>",
            ]

            page_lines = [
                "<html>",
                *header_lines,
                *body_lines,
                *footer_lines,
                "</html>",
            ]
            rl_path = self._get_list_index_page_path(rl_name)
            with open(rl_path, "w+") as f:
                for line in page_lines:
                    f.write(line + "\n")

    def build_target_pages(self, target_lookup, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for target_id, target in target_lookup.items():
            target_page_path = self._get_target_page_path(target_id)
            target_page_url = self._get_target_page_url(target_id)
            if target.updated or (not target_page_path.exists()):

                self.build_page_for_target(target, target_page_path, t_ref=t_ref)

                # write redirect pages...
                alt_ids = set(target.alt_ids.values())
                for alt_id in alt_ids:
                    if alt_id == target_id:
                        continue

                    redirect_page_path = self._get_target_page_path(alt_id)

                    self.build_redirect_page(redirect_page_path, target_page_path)

    def build_page_for_target(self, target: Target, target_page_path: Path, t_ref=None):
        t_ref = t_ref or Time.now()
        t_str = t_ref.strftime("%a, %d-%b-%Y, %H:%M")

        target_id = target.target_id

        header_lines = [f"<header>", f"<h1>{target_id}</h1>", "</header>"]

        target_info = target.get_info_lines(t_ref)
        body_lines = []
        for line in target_info:
            body_lines.append(f"<p>{line}</p>")

        if target.lc_fig_path is not None:
            www_fig_path = self.im_path / f"{target_id}_lc.png"
            shutil.copy2(target.lc_fig_path, www_fig_path)
            www_fig_url = www_fig_path.relative_to(self.local_www_path)
            body_lines.append(
                f"<img src='../im/{www_fig_path.name}' alt='{target_id} LC'>"
            )

        body_lines.append("</body>")
        footer_lines = [
            "<footer>",
            f"<p><small>Page updated {t_str}</small></p>",
            "</footer>",
        ]
        page_lines = [
            "<html>",
            *header_lines,
            *body_lines,
            *footer_lines,
            "</html>",
        ]
        with open(target_page_path, "w+") as f:
            for line in page_lines:
                f.write(line + "\n")

    def build_redirect_page(self, redirect_page_path, target_page_path):

        redirect_page_lines = [
            f"<html>",
            f"<head>",
            f"<meta http-equiv='refresh' content='0, url={target_page_path.name}'>",
            f"</head>",
            f"<body>Redirect: <a href='{target_page_path.name}'></a></body>" f"</html>",
        ]

        with open(redirect_page_path, "w+") as f:
            for line in redirect_page_lines:
                f.write(line + "\n")

    def clean_up_commit_history(self, target_lookup, t_ref: Time = None):
        t_ref = t_ref or Time.now()

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
        print(f"choose {ssh_auth_sock}")
        print("new env var:", os.environ["SSH_AUTH_SOCK"])

        ssh_auth_sock_path = Path(ssh_auth_sock)
        if ssh_auth_sock_path.exists():
            logger.info("newly set $SSH_AUTH_SOCK exists!")
        else:
            msg = "something went wrong - new $SSH_AUTH_SOCK does not exist..."
            logger.warning(msg)

    def publish(self, t_ref=None):
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
