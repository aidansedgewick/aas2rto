import os
import shutil
import subprocess
from logging import getLogger
from pathlib import Path

from astropy import units as u
from astropy.time import Time

from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


class GitWebpageManager:

    def __init__(
        self, www_config: dict, local_www_path: Path, target_lookup: TargetLookup
    ):
        self.www_config = www_config

        self.check_git_config()

        if local_www_path is None:
            msg = (
                f"In config, if `git_web: use: True`, must provide "
                f"Must provide path for 'www_path' in config_file/paths !"
            )
            raise ValueError(msg)

        self.local_www_path = Path(local_www_path).absolute()
        self.prepare_git_repo()

        self.target_lookup = target_lookup

        self.index_path = self.local_www_path / "index.html"

        self.im_path = self.local_www_path / "im"
        self.im_path.mkdir(exist_ok=True, parents=True)
        self.list_pages_path = self.local_www_path / "list"
        self.list_pages_path.mkdir(exist_ok=True, parents=True)
        self.target_pages_path = self.local_www_path / "target"
        self.target_pages_path.mkdir(exist_ok=True, parents=True)

    def check_git_config(self):
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

        if ranked_lists is None:
            logger.info("ranked_lists is None!")
            return

        for rl_name, ranked_list in ranked_lists.items():
            header_lines = [f"<header><h1>targets for {rl_name}</h1></header>"]

            col_names = (
                "rank name RA Dec. hhmmss ddmmss last_mag band last_obs dt0".split()
            )

            table_lines = ["<table>"]
            header_row = [
                "<tr>",
                " ".join(f"<td>{x}</td>" for x in col_names),
                "</tr>",
            ]
            table_lines.extend(header_row)

            if ranked_list is None:
                logger.warning(f"no ranked list for {rl_name}!")
                continue

            for ii, row in ranked_list.iterrows():
                target_id = row["target_id"]

                table_data = self.table_data_from_target_id(target_id, t_ref=t_ref)

                table_row = ["<tr>", f"<td>{ii+1}</td>"]
                for dat in table_data:
                    table_row.append(f"<td>{dat}</td>")

                table_row.append("</tr>")
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

    def table_data_from_target_id(self, target_id, t_ref: Time = None):
        t_Ref = t_ref or Time.now()

        target_page_path = self._get_target_page_path(target_id)

        target = self.target_lookup.get(target_id)

        lc = target.compiled_lightcurve
        last_mag, last_band, last_mjd = "-", "-", "-"
        if lc is not None:
            valid = lc[lc["tag"] == "valid"]
            if len(valid) > 0:
                last_mag = valid.iloc[-1]["mag"]
                last_band = valid.iloc[-1]["band"]
                last_mjd = Time(valid.iloc[-1]["mjd"], format="mjd")

        model = target.models.get("sncosmo_salt", None)
        if model is not None:
            t0 = model["t0"]
            dt = t_ref.mjd - t0
            dt_str = f"{dt:.1f}d"  #
        else:
            dt_str = "-"

        table_data = [
            f"<a href='../target/{target_page_path.name}'>{target_id}</a>",
            f"{target.ra:.5f}",
            f"{target.dec:+.4f}",
            f"{target.coord.ra.to_string(u.hour, precision=2, pad=True)}",
            f"{target.coord.dec.to_string(u.degree, alwayssign=True, precision=2, pad=True)}",
            f"{last_mag:.2f}",
            f"{last_band}",
            f"{last_mjd.iso}",
            f"{dt_str}",
        ]
        return table_data

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

        # target_info = target.get_info_lines(t_ref)
        info_lines = []

        info_lines.append("<h2>links</h2>")
        broker_name = target.alt_ids.get("ztf", None)
        if broker_name is not None:
            broker_lines = [
                f"    FINK: <a href='http://fink-portal.org/{broker_name}'>{broker_name}</a>",
                f"    Lasair: <a href='http://lasair-ztf.lsst.ac.uk/objects/{broker_name}'>{broker_name}</a>",
                f"    ALeRCE: <a href='http://alerce.online/object/{broker_name}'>{broker_name}</a>",
            ]
            info_lines.extend(broker_lines)

        tns_name = target.alt_ids.get("tns", None)
        if tns_name is not None:
            tns_url = f"http://wis-tns.org/object/{tns_name}"
            info_lines.append(f"    TNS: <a href='{tns_url}'>{tns_name}</a>")

        yse_name = target.alt_ids.get("yse", None)
        if yse_name is not None:
            yse_url = f"http://ziggy.ucolick.org/yse/transient_detail/{yse_name}"
            info_lines.append(f"    YSE: <a href='{yse_url}'>{yse_name}</a>")

        alt_rev = {}
        for source, alt_name in target.alt_ids.items():
            if alt_name not in alt_rev:
                alt_rev[alt_name] = [source]
            else:
                alt_rev[alt_name].append(source)

        info_lines.append("<h2>alt names</h2>")
        for name, source_list in alt_rev.items():
            l = f"    {name} (" + ",".join(source_list) + ")"
            info_lines.append(l)

        info_lines.append("<h2>coordinates</h2>")
        if target.ra is not None and target.dec is not None:
            eq_line = f"    equatorial (ra, dec) = ({target.ra:.4f},{target.dec:+.5f})"
            info_lines.append(eq_line)
        if target.coord is not None:
            gal = target.coord.galactic
            gal_line = f"    galactic (l, b) = ({gal.l.deg:.4f},{gal.b.deg:+.5f})"
            info_lines.append(gal_line)

        body_lines = []
        for line in info_lines:
            body_lines.append(f"<p>{line}</p>")

        if target.lc_fig_path is not None:
            www_fig_path = self.im_path / f"{target_id}_lc.png"
            shutil.copy2(target.lc_fig_path, www_fig_path)
            www_fig_url = www_fig_path.relative_to(self.local_www_path)
            body_lines.append(
                f"<p><img src='../im/{www_fig_path.name}' alt='{target_id} LC'></p>"
            )

        for fig_name, fig_path in target.additional_fig_paths.items():
            www_extra_fig_path = self.im_path / f"{target_id}_{fig_name}.png"
            shutil.copy2(fig_path, www_extra_fig_path)
            www_fig_url = www_fig_path.relative_to(self.local_www_path)
            body_lines.append(
                f"<p><img src='../im/{www_extra_fig_path.name}' alt='{target_id} {fig_name}'></p>"
            )

        phot_lines = []
        if target.compiled_lightcurve is not None:
            phot_lines.append("<h2>photometry</h2>")
            ndet = {}
            last_mag = {}
            if "band" in target.compiled_lightcurve.columns:
                for band, band_history in target.compiled_lightcurve.groupby("band"):
                    if "tag" in band_history.columns:
                        detections = band_history[band_history["tag"] == "valid"]
                    else:
                        detections = band_history
                    if len(detections) > 0:
                        ndet[band] = len(detections)
                        last_mag[band] = detections["mag"].iloc[-1]
            if len(last_mag) > 0:
                magvals_str = ", ".join(f"{k}={v:.2f}" for k, v in last_mag.items())
                mag_line = "    last " + magvals_str
                phot_lines.append(mag_line)
            if len(ndet) > 0:
                l = (
                    "    "
                    + ", ".join(f"{v} {k}" for k, v in ndet.items())
                    + " detections"
                )
                phot_lines.append(l)

        for line in phot_lines:
            body_lines.append(f"<p>{line}</p>")

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
