from logging import getLogger
from pathlib import Path

from astropy.time import Time

from aas2rto import paths
from aas2rto import utils

logger = getLogger(__name__.split(".")[-1])


class PathManager:

    default_base_path = paths.wkdir

    default_data_dir = "data"
    default_outputs_dir = "outputs"
    default_opp_targets_dir = "opp_targets"
    default_scratch_dir = "scratch"

    default_config = {
        "base_path": default_base_path,
        "project_path": "default",
        "project_base": "default",
        "paths": {},
    }

    def __init__(self, config: dict, create_paths=True):

        self.config = self.default_config.copy()
        self.config.update(config.copy())
        utils.check_unexpected_config_keys(
            self.config, self.default_config, name="path_manager", raise_exc=True
        )

        self.paths_config = self.config["paths"]
        self.process_paths(create_paths=create_paths)

    def process_paths(self, create_paths=True):
        base_path = self.config["base_path"]
        self.base_path = Path(base_path)

        project_path = self.config["project_path"] or "default"
        project_name = self.config["project_base"] or "default"
        if project_path == "default":
            if project_name == "default":
                msg = (
                    "You can set the name of the project_path by providing "
                    "'project_name' in 'paths:'. set to 'default'"
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

        self.lookup = {"base": self.base_path, "project": self.project_path}

        for location, raw_path in self.paths_config.items():
            if Path(raw_path).is_absolute():
                formatted_path = Path(raw_path)
            else:
                parts = str(raw_path).split("/")
                if parts[0].startswith("$"):
                    # eg. replace `$my_cool_dir/blah/blah` with `lookup["my_cool_dir"]`
                    parent_name = parts[0][1:]
                    parent = self.lookup[parent_name]
                    formatted_parts = [Path(p) for p in parts[1:]]
                else:
                    parent = self.project_path
                    formatted_parts = [Path(p) for p in parts]
                formatted_path = parent.joinpath(*formatted_parts)
            self.lookup[location] = formatted_path

        if "data" not in self.lookup:
            self.lookup["data"] = self.project_path / "data"
        self.data_path = self.lookup["data"]

        if "outputs" not in self.lookup:
            self.lookup["outputs"] = self.project_path / "outputs"
        self.outputs_path = self.lookup["outputs"]

        if "opp_targets" not in self.lookup:
            self.lookup["opp_targets"] = self.project_path / "opp_targets"
        self.opp_targets_path = self.lookup["opp_targets"]

        if "scratch" not in self.lookup:
            self.lookup["scratch"] = self.project_path / "scratch"
        self.scratch_path = self.lookup["scratch"]

        if "comments" not in self.lookup:
            self.lookup["comments"] = self.project_path / "comments"
        self.comments_path = self.lookup["comments"]

        if "rejected_targets" not in self.lookup:
            self.lookup["rejected_targets"] = self.project_path / "rejected_targets"
        self.rejected_targets_path = self.lookup["rejected_targets"]

        if "recovery" not in self.lookup:
            self.lookup["recovery"] = self.project_path / "recovery_files"
        self.recovery_path = self.lookup["recovery"]

        if create_paths:
            self.create_paths()

    def create_paths(self):
        for path_name, path_val in self.lookup.items():
            path_val.mkdir(exist_ok=True, parents=True)

    def get_output_plots_path(self, sub_dir, mkdir=True) -> Path:
        plots_path = self.outputs_path / f"plots/{sub_dir}"
        if mkdir:
            plots_path.mkdir(exist_ok=True, parents=True)
        return plots_path

    def get_visible_targets_list_path(self, list_stem, mkdir=True) -> Path:
        visible_targets_path = self.outputs_path / "visible_targets"
        if mkdir:
            visible_targets_path.mkdir(exist_ok=True, parents=True)
        return visible_targets_path / f"{list_stem}.csv"

    def get_ranked_list_path(self, list_stem, mkdir=True) -> Path:
        ranked_lists_path = self.outputs_path / "ranked_lists"
        if mkdir:
            ranked_lists_path.mkdir(exist_ok=True, parents=True)
        return ranked_lists_path / f"{list_stem}.csv"

    def get_lightcurve_plot_path(self, target_id, mkdir=True):
        scratch_lc_path = self.scratch_path / "lc"
        if mkdir:
            scratch_lc_path.mkdir(exist_ok=True, parents=True)
        return scratch_lc_path / f"{target_id}_lc.png"

    def get_visibility_plot_path(self, target_id, obs_name, mkdir=True):
        scratch_vis_path = self.scratch_path / "vis"
        if mkdir:
            scratch_vis_path.mkdir(exist_ok=True, parents=True)
        return scratch_vis_path / f"{target_id}_{obs_name}_vis.png"

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
