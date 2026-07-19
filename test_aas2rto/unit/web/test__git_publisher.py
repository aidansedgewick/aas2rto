import pytest
import subprocess
from pathlib import Path

from aas2rto.exc import MissingKeysError, UnexpectedKeysError
from aas2rto.web.publishers.git_publisher import GitPublisher


@pytest.fixture
def git_publisher_config(tmp_path: Path):

    deploy_key_path = tmp_path / "test_deploy_key"
    deploy_key_path.touch()
    return {
        "user_email": "user@example.com",
        "remote_url": "git@github.com/example/example",
        "remote_name": "origin",
        "branch": "some_branch",
        "deploy_key_path": deploy_key_path,
    }


@pytest.fixture
def web_base_path(tmp_path: Path):
    p = tmp_path / "static"
    p.mkdir()
    return p


@pytest.fixture
def patched_publisher(git_publisher_config: dict, web_base_path: Path):
    return GitPublisher(git_publisher_config, web_base_path)


class Test__InitPublisher:
    def test__unexpected_git_key_raises(self, git_publisher_config: dict, tmp_path):
        # Arrange
        git_publisher_config["aaagh"] = "some text here"

        # Act
        with pytest.raises(UnexpectedKeysError):
            p = GitPublisher(git_publisher_config, tmp_path)

    def test__call_init_new_repo(
        self,
        git_publisher_config: dict,
        web_base_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        def cause_exception(*args, **kwargs):
            raise subprocess.CalledProcessError(404, "clone")

        monkeypatch.setattr(GitPublisher, "clone_existing_repo", cause_exception)

        # Act
        publisher = GitPublisher(git_publisher_config, web_base_path)

        # Assert
        assert publisher.git_repo_status == "new_init"

    def test__use_existing_repo(self, git_publisher_config: dict, web_base_path: Path):
        # Arrange
        git_dir = web_base_path / ".git"
        git_dir.mkdir(exist_ok=True, parents=True)

        # Act
        publisher = GitPublisher(git_publisher_config, web_base_path)

        # Assert
        assert publisher.git_repo_status == "existing"

    def test__bad_config_key_raises(
        self, git_publisher_config: dict, web_base_path: Path
    ):
        # Arrange
        git_publisher_config["blah"] = 100.0

        # Act
        with pytest.raises(UnexpectedKeysError):
            publisher = GitPublisher(git_publisher_config, web_base_path)

    def test__missing_config_key_raises(
        self, git_publisher_config: dict, web_base_path: Path
    ):
        # Arrange
        git_publisher_config.pop("user_email")

        # Act
        with pytest.raises(MissingKeysError):
            publisher = GitPublisher(git_publisher_config, web_base_path)
