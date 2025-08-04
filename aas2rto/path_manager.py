from logging import getLogger
from pathlib import Path

from astropy.time import Time

from aas2rto import paths

logger = getLogger(__name__.split(".")[-1])


class PathManager:

    default_base_path = paths.wkdir

    def __init__(self, paths_config, project_name=None, create_paths=True):
        self.paths_config = paths_config.copy()
        self.process_paths(project_name=None, create_paths=create_paths)

    def process_paths(self, project_name=None, create_paths=True):
        base_path = self.paths_config.pop("base_path", "default")
        if base_path == "default":
            base_path = self.default_base_path
        self.base_path = Path(base_path)

        project_path = self.paths_config.pop("project_path", None)
        if project_path is None or project_path == "default":
            if project_name is None:
                msg = (
                    "You can set the name of the project_path by providing "
                    "'project_name' in 'selector_parameters'. set to 'default'"
                )
                logger.info(msg)
                project_name = "default"
            projects_base = self.base_path / "projects"
            projects_base.mkdir(exist_ok=True, parents=True)
            project_path = projects_base / project_name
        self.project_path = Path(project_path)

        msg = (
            f"set project path at:\n    \033[36;1m{self.project_path.absolute()}\033[0m"
        )
        logger.info(msg)

        self.lookup = {"base_path": self.base_path, "project_path": self.project_path}

        for location, raw_path in self.paths_config.items():
            if Path(raw_path).is_absolute():
                formatted_path = Path(raw_path)
            else:
                parts = str(raw_path).split("/")
                if parts[0].startswith("$"):
                    # eg. replace `$my_cool_dir/blah/blah` with `paths["my_cool_dir"]`
                    parent_name = parts[0][1:]
                    parent = self.lookup[parent_name]
                    formatted_parts = [Path(p) for p in parts[1:]]
                else:
                    parent = self.project_path
                    formatted_parts = [Path(p) for p in parts]
                formatted_path = parent.joinpath(*formatted_parts)
            self.lookup[location] = formatted_path

        if "data_path" not in self.lookup:
            self.lookup["data_path"] = self.project_path / paths.default_data_dir
        self.data_path = self.lookup["data_path"]

        if "outputs_path" not in self.lookup:
            self.lookup["outputs_path"] = self.project_path / paths.default_outputs_dir
        self.outputs_path = self.lookup["outputs_path"]

        if "opp_targets_path" not in self.lookup:
            self.lookup["opp_targets_path"] = (
                self.project_path / paths.default_opp_targets_dir
            )
        self.opp_targets_path = self.lookup["opp_targets_path"]

        if "scratch_path" not in self.lookup:
            self.lookup["scratch_path"] = self.project_path / paths.default_scratch_dir
        self.scratch_path = self.lookup["scratch_path"]
        self.lc_scratch_path = self.scratch_path / "lc"
        self.lookup["lc_scratch_path"] = self.lc_scratch_path
        self.vis_scratch_path = self.scratch_path / "vis"
        self.lookup["vis_scratch_path"] = self.vis_scratch_path

        if "comments_path" not in self.lookup:
            self.lookup["comments_path"] = self.project_path / "comments"
        self.comments_path = self.lookup["comments_path"]

        if "rejected_targets_path" not in self.lookup:
            self.lookup["rejected_targets_path"] = (
                self.project_path / "rejected_targets"
            )
        self.rejected_targets_path = self.lookup["rejected_targets_path"]

        if "recovery_path" not in self.lookup:
            self.lookup["recovery_path"] = self.project_path / "recovery_files"
        self.recovery_path = self.lookup["recovery_path"]

        if create_paths:
            for path_name, path_val in self.lookup.items():
                path_val.mkdir(exist_ok=True, parents=True)

    def get_rank_history_file(self, target_id: str):
        rank_history_path = self.paths.get("rank_history_path", None)
        if rank_history_path is None:
            rank_history_path = self.recovery_path / "rank_history"
        rank_history_path.mkdir(exist_ok=True, parents=True)
        return rank_history_path / f"{target_id}.csv"

    def get_score_history_file(self, target_id: str):
        score_history_path = self.lookup.get("score_history_path", None)
        if score_history_path is None:
            score_history_path = self.recovery_path / "score_history"
        score_history_path.mkdir(exist_ok=True, parents=True)
        return score_history_path / f"{target_id}.csv"

    def get_output_plots_path(self, obs_name, mkdir=True) -> Path:
        plots_path = self.outputs_path / f"plots/{obs_name}"
        if mkdir:
            plots_path.mkdir(exist_ok=True, parents=True)
        return plots_path

    def get_visible_targets_list_path(self, obs_name, mkdir=True) -> Path:
        visible_targets_path = self.outputs_path / "visible_targets"
        if mkdir:
            visible_targets_path.mkdir(exist_ok=True, parents=True)
        return visible_targets_path / f"{obs_name}.csv"

    def get_ranked_list_path(self, obs_name, mkdir=True) -> Path:
        ranked_lists_path = self.outputs_path / "ranked_lists"
        if mkdir:
            ranked_lists_path.mkdir(exist_ok=True, parents=True)
        return ranked_lists_path / f"{obs_name}.csv"

    def get_lightcurve_plot_path(self, target_id):
        return self.lc_scratch_path / f"{target_id}_lc.png"

    def get_visibility_plot_path(self, target_id, obs_name):
        return self.vis_scratch_path / f"{target_id}_{obs_name}_vis.png"

    def get_current_recovery_file(self, stem="recover", t_ref=None, fmt="json"):
        t_ref = t_ref or Time.now()

        timestamp = t_ref.strftime("%y%m%d_%H%M%S")
        filename = f"{stem}_{timestamp}"
        return self.recovery_path / f"{filename}.{fmt}"

    def get_existing_recovery_files(self, stem="recover", fmt="json"):
        return sorted(self.recovery_path.glob(f"{stem}*.{fmt}"))

    def get_current_rank_history_file(
        self, stem="rank_history", t_ref=None, fmt="json"
    ):
        t_ref = t_ref or Time.now()

        timestamp = t_ref.strftime("%y%m%d_%H%M%S")
        filename = f"{stem}_{timestamp}"
        return self.recovery_path / f"{filename}.{fmt}"

    def get_existing_rank_history_files(self, stem="rank_history", fmt="json"):
        return sorted(self.recovery_path.glob(f"{stem}*.{fmt}"))
