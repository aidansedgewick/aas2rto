import pytest
import subprocess

from astropy.time import Time

from aas2rto.exc import MissingKeysError, UnexpectedKeysError
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager

from aas2rto.web.static_pages_manager import StaticPagesManager


# dummy_subprocess defined in unit/conftest.py


@pytest.fixture
def empty_web_config():
    return {}


@pytest.fixture
def static_pages_config():
    return {
        "git": {
            "user_email": "user@example.com",
            "remote_url": "git@github.com/example/example",
            "remote_name": "origin",
            "branch": "some_branch",
        },
        "publish_interval": 3600.0,
    }


@pytest.fixture
def prepared_outputs_mgr(outputs_mgr_with_plots: OutputsManager, t_fixed: Time):
    outputs_mgr = outputs_mgr_with_plots  # short name is nicer...
    outputs_mgr.observatory_manager.apply_ephem_info(t_fixed)
    outputs_mgr.build_ranked_target_lists(plots=False, write_list=False)
    outputs_mgr.build_visible_target_lists(plots=False, write_list=False)
    T00 = outputs_mgr.target_lookup["T00"]
    T00.update_science_score_history(1.0, t_fixed)
    T00.updated = True
    T00.additional_fig_paths["extra_fig"] = T00.lc_fig_path
    return outputs_mgr


@pytest.fixture
def page_mgr(
    static_pages_config: dict,
    prepared_outputs_mgr: OutputsManager,
    path_mgr: PathManager,
) -> StaticPagesManager:

    return StaticPagesManager(static_pages_config, prepared_outputs_mgr, path_mgr)


class Test__InitStaticPages:

    def test__empty_config(
        self, prepared_outputs_mgr: OutputsManager, path_mgr: PathManager
    ):
        # Act
        sp_mgr = StaticPagesManager({}, prepared_outputs_mgr, path_mgr)

        # Assert
        assert sp_mgr.git_config is None
        exp_web_path = path_mgr.project_path / "web/static"

        assert sp_mgr.web_base_path == exp_web_path
        assert (sp_mgr.web_base_path / "im").exists()
        assert (sp_mgr.web_base_path / "lists").exists()
        assert (sp_mgr.web_base_path / "target").exists()

    def test__with_config(
        self,
        static_pages_config: dict,
        prepared_outputs_mgr: OutputsManager,
        path_mgr: PathManager,
    ):
        # Act
        sp_mgr = StaticPagesManager(static_pages_config, prepared_outputs_mgr, path_mgr)

        # Assert
        isinstance(sp_mgr.git_config, dict)
        assert set(sp_mgr.config.keys()) == set(["git", "publish_interval"])

        assert sp_mgr.git_repo_status == "cloned"

    def test__bad_config_key_raises(
        self,
        static_pages_config: dict,
        prepared_outputs_mgr: OutputsManager,
        path_mgr: PathManager,
    ):
        # Arrange
        static_pages_config["blah"] = 100.0

        # Act
        with pytest.raises(UnexpectedKeysError):
            mgr = StaticPagesManager(
                static_pages_config, prepared_outputs_mgr, path_mgr
            )

    def test__missing_git_key_raises(
        self,
        static_pages_config: dict,
        prepared_outputs_mgr: OutputsManager,
        path_mgr: PathManager,
    ):
        # Arrange
        static_pages_config["git"].pop("user_email")

        # Act
        with pytest.raises(MissingKeysError):
            mgr = StaticPagesManager(
                static_pages_config, prepared_outputs_mgr, path_mgr
            )

    def test__unexpected_git_key_raises(
        self,
        static_pages_config: dict,
        prepared_outputs_mgr: OutputsManager,
        path_mgr: PathManager,
    ):
        # Arrange
        static_pages_config["git"]["aaagh"] = "some text here"

        # Act
        with pytest.raises(UnexpectedKeysError):
            mgr = StaticPagesManager(
                static_pages_config, prepared_outputs_mgr, path_mgr
            )

    def test__call_init_new_repo(
        self,
        static_pages_config: dict,
        prepared_outputs_mgr: OutputsManager,
        path_mgr: PathManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        def cause_exception(*args, **kwargs):
            raise subprocess.CalledProcessError(404, "clone")

        monkeypatch.setattr(StaticPagesManager, "clone_existing_repo", cause_exception)

        # Act
        mgr = StaticPagesManager(static_pages_config, prepared_outputs_mgr, path_mgr)

        # Assert
        assert mgr.git_repo_status == "new_init"

    def test__use_existing_repo(
        self,
        static_pages_config: dict,
        prepared_outputs_mgr: OutputsManager,
        path_mgr: PathManager,
    ):
        # Arrange
        git_dir = path_mgr.web_path / "static/.git"
        git_dir.mkdir(exist_ok=True, parents=True)

        # Act
        mgr = StaticPagesManager(static_pages_config, prepared_outputs_mgr, path_mgr)

        # Assert
        assert mgr.git_repo_status == "existing"


class Test__BuildWebpageTarget:
    def test__build_target_webpage(self, page_mgr: StaticPagesManager):
        # Arrange
        T00 = page_mgr.outputs_manager.target_lookup["T00"]

        # Act
        page_mgr.build_webpage_for_target(T00)

        # Assert
        targets_path = page_mgr.path_manager.project_path / "web/static/target"
        main_exp_path = targets_path / "T00.html"
        assert main_exp_path.exists()
        # Redirect pages NOT written here - but in main "write_webpages loop"

    def test__build_redirect_pages(self, page_mgr: StaticPagesManager):
        # Arrange
        T00 = page_mgr.outputs_manager.target_lookup["T00"]

        # Act
        page_mgr.build_redirect_pages_for_target(T00)

        # Assert
        targets_path = page_mgr.path_manager.project_path / "web/static/target"
        exp_path = targets_path / "target_A.html"
        assert exp_path.exists()

    def test__build_index_page(self, page_mgr: StaticPagesManager):
        # Act
        page_mgr.build_index_page()

        # Assert
        assert page_mgr.index_path.exists()

    def test__build_science_ranked_page(self, page_mgr: StaticPagesManager):
        # Act
        page_mgr.build_science_ranked_page()

        # Assert
        lists_path = page_mgr.path_manager.project_path / "web/static/lists"
        exp_path = lists_path / "sci_ranked.html"
        assert exp_path.exists()

    def test__build_obs_ranked_list(self, page_mgr: StaticPagesManager):
        # Act
        page_mgr.build_obs_ranked_pages()

        # Assert
        lists_path = page_mgr.path_manager.project_path / "web/static/lists"
        lasilla_list = lists_path / "lasilla_ranked.html"
        lasilla_list.exists()
        astrolab_list = lists_path / "astrolab_ranked.html"
        astrolab_list.exists()

    def test__build_obs_visible_list(self, page_mgr: StaticPagesManager):
        # Act
        page_mgr.build_obs_visible_pages()

        # Assert
        lists_path = page_mgr.path_manager.project_path / "web/static/lists"
        lasilla_list = lists_path / "lasilla_visible.html"
        lasilla_list.exists()
        astrolab_list = lists_path / "astrolab_visible.html"
        astrolab_list.exists()

    def test__build_all_webpages(self, page_mgr: StaticPagesManager):
        # Act
        page_mgr.build_webpages()

        # Assert
        lists_path = page_mgr.path_manager.project_path / "web/static/lists"
        target_path = page_mgr.path_manager.project_path / "web/static/target"

        lists_written = [f.stem for f in lists_path.glob("*.html")]
        exp_lists = [
            "astrolab_ranked",
            "astrolab_visible",
            "lasilla_ranked",
            "lasilla_visible",
            "sci_ranked",
        ]
        assert set(lists_written) == set(exp_lists)
        targets_written = [f.stem for f in target_path.glob("*.html")]
        assert set(targets_written) == set(["T00", "T01", "target_A", "target_B"])
        # check page written for missing T01, even if not updated...
        # also check redirect pages are written


class Test__PublishToGit:
    def test__publish_to_git(
        self, page_mgr: StaticPagesManager, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange
        def refresh_pass(*args, **kwargs):
            pass

        monkeypatch.setattr(page_mgr, "refresh_tmux_ssh_auth_sock", refresh_pass)

        # Act
        page_mgr.publish_to_git()  # subprocess.check_output auto-patched in conftest...

        # Assert
        # effectively checking for typos here...


class Test__PerformAllTasks:
    def test__perform_all_tasks(self, page_mgr: StaticPagesManager):

        # Act
        page_mgr.perform_all_tasks()
