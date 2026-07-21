from __future__ import annotations

import hashlib
import json
import os
import requests
import shutil
import subprocess
import time
from logging import getLogger
from pathlib import Path
from typing import Protocol

import tqdm

import pandas as pd

from jinja2 import Environment, FileSystemLoader

from astropy.time import Time

from aas2rto import utils
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.utils import QueryTracker
from aas2rto.web.publishers.git_publisher import GitPublisher
from aas2rto.web.publishers.vercel_publisher import VercelPublisher

logger = getLogger(__name__.split(".")[-1])


def h2_header_formatter(text: str):
    return f"<h2>{text}</h2>"


class PublisherProtocol(Protocol):
    def publish(self) -> None:
        pass


class StaticPagesManager:

    default_config = {
        "publish_interval": 1800.0,
        "git": None,
        "vercel": None,
    }

    def __init__(
        self, config: dict, outputs_manager: OutputsManager, path_manager: PathManager
    ):
        logger.info("init static pages manager")
        self.config = self.default_config.copy()
        self.config.update(config)
        utils.check_unexpected_config_keys(
            self.config, self.default_config, name="web.static_pages", raise_exc=True
        )

        self.outputs_manager = outputs_manager
        self.path_manager = path_manager

        self.web_base_path = self.path_manager.web_path / "static"
        self.index_path = self.web_base_path / "index.html"
        self.web_im_path = self.web_base_path / "im"
        self.web_lists_path = self.web_base_path / "lists"
        self.web_target_path = self.web_base_path / "target"

        self.initialize_publishers()

        self.create_web_directories()  # Do this last - allow for possible clone.

        self.web_environment = Environment(
            loader=FileSystemLoader(self.path_manager.web_templates_path)
        )
        self.last_git_publish = 0.0
        self.last_vercel_publish = 0.0

    def create_web_directories(self):
        self.web_base_path.mkdir(exist_ok=True, parents=True)
        self.web_im_path.mkdir(exist_ok=True, parents=True)

        self.web_lists_path.mkdir(exist_ok=True, parents=True)

        self.web_target_path = self.web_base_path / "target"
        self.web_target_path.mkdir(exist_ok=True, parents=True)

    def initialize_publishers(self):
        self.publishers: dict[str, PublisherProtocol] = {}

        git_config: dict = self.config["git"] or {}  # .get(None, {}) gives 'None'...
        use_git = git_config.get("use", True)  # ...so this line failes.
        if git_config and use_git:
            self.publishers["git"] = GitPublisher(git_config, self.web_base_path)
        else:
            logger.info("not using git publisher")

        vercel_config: dict = self.config["vercel"] or {}
        use_vercel = vercel_config.get("use", True)
        if vercel_config and use_vercel:
            self.publishers["vercel"] = VercelPublisher(
                vercel_config, self.web_base_path
            )
        else:
            logger.info("not using vercel")

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

        target_pages_built = 0
        for target_id, target in self.outputs_manager.target_lookup.items():
            target_page_path = self._get_target_page_path(target_id)

            page_age = utils.calc_file_age(target_page_path, t_ref=t_ref)  # in DAYS
            page_old = page_age > 1.0

            if not target_page_path.exists() or target.updated or page_old:
                self.build_webpage_for_target(target, t_ref=t_ref)
                self.build_redirect_pages_for_target(target)
                target_pages_built = target_pages_built + 1
        if target_pages_built > 0:
            logger.info(f"built target pages for {target_pages_built}")

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

            data = []
            for key, val in row.items():
                if key == "target_id":
                    continue
                if isinstance(val, bool):
                    row_val = "\u2713" if val else ""
                if isinstance(val, float):
                    row_val = f"{val:.2e}"
                else:
                    row_val = val

                data.append(row_val)

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
            additional_path_list.append(im_path)

        target_info_lines = target.get_info_lines(
            header_formatter=h2_header_formatter,
            link_formatter=utils.format_link_as_html,
        )

        page_data = dict(
            target_id=target.target_id,
            target_info_lines=target_info_lines,
            lc_fig_path=rel_lc_path,
            vis_fig_path_list=rel_vis_paths,
            additional_fig_path_list=additional_path_list,
            index_url="../index.html",  # need relative link....
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
            redirect_url = f"../{self._get_target_page_url(target.target_id)}"  # REL!
            page_data = dict(
                redirect_url=redirect_url,
                target_id=target.target_id,
            )
            rendered_page = template.render(page_data)
            redirect_page_path = self._get_target_page_path(alt_id)
            with open(redirect_page_path, "w+") as f:
                f.writelines(rendered_page)

    def publish(self):
        for publisher_name, publisher in self.publishers.items():
            publisher.publish()

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        self.build_webpages(t_ref=t_ref)
        self.publish()
