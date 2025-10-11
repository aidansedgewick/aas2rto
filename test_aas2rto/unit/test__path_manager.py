import pytest
from pathlib import Path

from aas2rto.path_manager import PathManager

from astropy.time import Time


@pytest.fixture
def basic_config(tmp_path):
    return {"base_path": tmp_path}


@pytest.fixture
def basic_pm(basic_config):
    basic_config["project_name"] = "test"
    return PathManager(basic_config)


@pytest.fixture
def project_path(tmp_path):
    return tmp_path / "projects/test"


class Test__PathManagerInit:
    def test__basic_config(self, basic_config: dict, tmp_path: Path):
        # Act
        pm = PathManager(basic_config)

        # Assert
        assert pm.base_path == tmp_path
        assert pm.project_path == tmp_path / "projects/default"
        assert pm.project_path.exists()

        expected_keys = [
            "base",
            "project",
            "data",
            "outputs",
            "opp_targets",
            "scratch",
            "comments",
            "rejected_targets",
            "recovery",
        ]
        assert set(pm.lookup.keys()) == set(expected_keys)

        assert pm.lookup["data"] == tmp_path / "projects/default/data"
        assert pm.lookup["outputs"] == tmp_path / "projects/default/outputs"
        assert pm.lookup["opp_targets"] == tmp_path / "projects/default/opp_targets"
        assert pm.lookup["scratch"] == tmp_path / "projects/default/scratch"
        assert pm.lookup["comments"] == tmp_path / "projects/default/comments"
        assert (
            pm.lookup["rejected_targets"]
            == tmp_path / "projects/default/rejected_targets"
        )
        assert pm.lookup["recovery"] == tmp_path / "projects/default/recovery_files"

    def test__custom_paths_processed(self, basic_config: dict, tmp_path: Path):
        # Arrange
        basic_config["paths"] = {}
        basic_config["paths"]["data"] = "sne_data"
        basic_config["paths"]["src01_data"] = "$data/src01"
        basic_config["paths"]["other"] = "other_stuff"

        # Act
        pm = PathManager(config=basic_config)

        # Assert
        # custom path overwrites default
        assert pm.lookup["data"] == tmp_path / "projects/default/sne_data"
        assert pm.lookup["data"].exists()
        # $-aliasing works...
        assert pm.lookup["src01_data"] == tmp_path / "projects/default/sne_data/src01"
        assert pm.lookup["src01_data"].exists()
        # new paths also created
        assert pm.lookup["other"] == tmp_path / "projects/default/other_stuff"
        assert pm.lookup["other"].exists()

    def test__custom_project_name(self, basic_config: dict, tmp_path: Path):
        # Arrange
        basic_config["project_name"] = "cool_project"

        # Act
        pm = PathManager(basic_config)

        # Assert
        assert pm.base_path == tmp_path
        assert pm.project_path == tmp_path / "projects/cool_project"

        assert pm.lookup["data"] == tmp_path / "projects/cool_project/data"


class Test__PathFunctions:
    def test__output_plots(self, basic_pm: PathManager, project_path: Path):
        # Act
        outpath = basic_pm.get_output_plots_path("astrolab")

        # Assert
        assert outpath == project_path / "outputs/plots/astrolab"
        assert outpath.is_dir()
        assert outpath.exists()

    def test__vis_targets_list(self, basic_pm: PathManager, project_path: Path):
        # Act
        outpath = basic_pm.get_visible_targets_list_path("astrolab")

        # Assert
        assert outpath == project_path / "outputs/visible_targets/astrolab.csv"

    def test__ranked_targets(self, basic_pm: PathManager, project_path: Path):
        # Act
        outpath = basic_pm.get_ranked_list_path("astrolab")

        # Assert
        assert outpath == project_path / "outputs/ranked_lists/astrolab.csv"

    def test__lc_plot(self, basic_pm: PathManager, project_path: Path):
        # Act
        outpath = basic_pm.get_lightcurve_plot_path("T000")

        # Assert
        assert outpath == project_path / "scratch/lc/T000_lc.png"

    def test__vis_plot(self, basic_pm: PathManager, project_path: Path):
        # Act
        outpath = basic_pm.get_visibility_plot_path("T000", "astrolab")

        # Assert
        assert outpath == project_path / "scratch/vis/T000_astrolab_vis.png"

    def test__get_current_recovery_file(
        self, basic_pm: PathManager, project_path: Path, t_fixed: Time
    ):
        # Act
        outpath = basic_pm.get_current_recovery_file(t_ref=t_fixed)

        # Assert
        rec_path = project_path / "recovery_files"
        assert outpath == rec_path / "recover_230225_000000.json"

    def test__get_existing_recovery_files(
        self, basic_pm: PathManager, project_path: Path
    ):
        # Arrange
        rec_path = project_path / "recovery_files"
        with open(rec_path / "recover_A.json", "w+") as f:
            f.write("0")
        with open(rec_path / "recover_B.csv", "w+") as f:
            f.write("0")
        with open(rec_path / "test_C.json", "w+") as f:
            f.write("0")
        with open(rec_path / "recover_D.json", "w+") as f:
            f.write("0")

        # Act
        files = basic_pm.get_existing_recovery_files()

        # Assert
        assert len(files) == 2  # ignore non-json, ignore incorrect stem.
        assert files[0] == rec_path / "recover_A.json"
        assert files[1] == rec_path / "recover_D.json"

    def test__get_current_rank_hist_file(
        self, basic_pm: PathManager, project_path: Path, t_fixed: Time
    ):
        # Act
        outpath = basic_pm.get_current_rank_history_file(t_ref=t_fixed)

        # Assert
        rec_path = project_path / "recovery_files"
        assert outpath == rec_path / "rank_history_230225_000000.json"

    def test__get_existing_rank_hist_files(
        self, basic_pm: PathManager, project_path: Path
    ):
        # Arrange
        rec_path = project_path / "recovery_files"
        with open(rec_path / "rank_history_A.json", "w+") as f:
            f.write("0")
        with open(rec_path / "rank_history_B.csv", "w+") as f:
            f.write("0")
        with open(rec_path / "test_C.json", "w+") as f:
            f.write("0")
        with open(rec_path / "rank_history_D.json", "w+") as f:
            f.write("0")

        # Act
        files = basic_pm.get_existing_rank_history_files()

        # Assert
        assert len(files) == 2  # ignore non-json, ignore incorrect stem.
        assert files[0] == rec_path / "rank_history_A.json"
        assert files[1] == rec_path / "rank_history_D.json"
