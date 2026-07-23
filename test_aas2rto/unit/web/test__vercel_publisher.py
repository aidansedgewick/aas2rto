import hashlib
import json
import pytest
import requests
from pathlib import Path

from aas2rto.exc import MissingKeysError, UnexpectedKeysError
from aas2rto.web.publishers.vercel_publisher import VercelPublisher


@pytest.fixture
def web_base_path(tmp_path: Path):
    web_base_path = tmp_path / "static"
    web_base_path.mkdir(exist_ok=True, parents=True)
    return web_base_path


@pytest.fixture
def vercel_config():
    return {
        "project_token": "test_token",
        "project_name": "test_project",
    }


class MockResponse(requests.Response):
    pass


@pytest.fixture(autouse=True)
def patch_requests_post(monkeypatch: pytest.MonkeyPatch):
    def mock_post(*args, data: bytes = None, **kwargs):
        resp = MockResponse()
        resp.status_code = 200
        resp.reason = "ok"
        if data is not None:
            data_str = data.decode() if isinstance(data, bytes) else str(data)
            if "fail" in data_str:
                resp.status_code = 400
                resp.reason = "Fail requested"
            resp._content = data

        return resp

    monkeypatch.setattr(requests.Session, "post", mock_post)


@pytest.fixture(autouse=True)
def patch_requests_get(monkeypatch: pytest.MonkeyPatch):
    def mock_get(self, url: str, **kwargs):
        resp = MockResponse()
        resp.status_code = 200
        resp.reason = "ok"
        if "fail" in url:
            resp.status_code = 404
            resp.reason = "Fail requested"

        return resp

    monkeypatch.setattr(requests.Session, "get", mock_get)


@pytest.fixture
def patched_publisher(
    vercel_config: dict,
    web_base_path: Path,
):
    publisher = VercelPublisher(vercel_config, web_base_path)
    publisher.vercel_client.vercel_api = "https://example.com"  # NEVER contact real api
    return publisher


##== Test fixtures work


class Test__PatchedRequests:
    def test__session(self, patch_requests_post: None):
        # Arrange
        session = requests.Session()

        # Act
        resp = session.post("blah", data="test_data".encode())

        # Assert
        assert isinstance(resp, MockResponse)

    def test__request_failure(self, patch_requests_post):
        # Act
        with requests.Session() as session:
            resp = session.post("blah", data="please fail".encode())

        # Assert
        assert resp.status_code == 400

    def test__mock_resp_raise_for_status(self, patch_requests_post):
        # Arrange
        with requests.Session() as session:
            resp = session.post("blah", data="please fail".encode())

        # Act
        with pytest.raises(requests.HTTPError):
            resp.raise_for_status()


##===== Real tests start here!


class Test__VercelInit:
    def test__init_normal(
        self, vercel_config: dict, web_base_path: Path, tmp_path: Path
    ):
        # Act
        vp = VercelPublisher(vercel_config, web_base_path)

        # Assert
        assert isinstance(vp.config, dict)
        assert vp.manifest_filepath == tmp_path / "vercel_manifest.json"

    def test__no_token_raises(self, vercel_config: dict, web_base_path: Path):
        # Arrange
        vercel_config.pop("project_token")

        # Act
        with pytest.raises(MissingKeysError):
            vp = VercelPublisher(vercel_config, web_base_path)

    def test__no_name_raises(self, vercel_config: dict, web_base_path: Path):
        # Arrange
        vercel_config.pop("project_name")

        # Act
        with pytest.raises(MissingKeysError):
            vp = VercelPublisher(vercel_config, web_base_path)


class Test__Manifest:
    def test__load_existing(self, patched_publisher: VercelPublisher):
        # Arrange
        test_data = {"x": 100, "y": 1000}
        with open(patched_publisher.manifest_filepath, "w+") as f:
            json.dump(test_data, f)

        # Act
        manifest = patched_publisher.load_manifest()

        # Assert
        assert set(manifest.keys()) == set(["x", "y"])

    def test__missing_loads_empty(self, patched_publisher: VercelPublisher):
        # Arrange
        assert not patched_publisher.manifest_filepath.exists()

        # Act
        manifest = patched_publisher.load_manifest()

        # Assert
        assert set(manifest) == set()

    def test__write_manifest(self, patched_publisher: VercelPublisher):
        # Arrange
        data = {"x": 100, "y": 1000}

        # Act
        patched_publisher.write_manifest(data)

        # Assert
        assert patched_publisher.manifest_filepath.exists()


class Test__GetUploads:

    def test__get_uploads(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["Line1"])

        # Act
        upload_payloads = patched_publisher.get_upload_payloads()

        # Assert
        assert isinstance(upload_payloads, dict)
        assert set(upload_payloads) == set(["file01.txt"])

        file01_payload = upload_payloads["file01.txt"]
        assert isinstance(file01_payload, dict)
        assert set(file01_payload.keys()) == set(["file", "sha", "size", "data"])
        assert isinstance(file01_payload["file"], str)
        assert isinstance(file01_payload["sha"], str)
        assert isinstance(file01_payload["size"], int)
        assert isinstance(file01_payload["data"], bytes)

    def test__skip_unmodified(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange
        # Write a file and make sure it's in the manifest
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["Line1"])
        payloads = patched_publisher.get_upload_payloads()
        payloads["file01.txt"].pop("data")
        patched_publisher.write_manifest(payloads)
        # Write the same again, and check it's not updated.
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["Line1"])

        # Act
        new_payloads = patched_publisher.get_upload_payloads()

        # Assert
        assert set(new_payloads) == set()  # Nothing to upload!

    def test__include_modified(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange

        # Write a file and make sure it's in the manifest
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["Line1"])
        payloads = patched_publisher.get_upload_payloads()
        payloads["file01.txt"].pop("data")
        patched_publisher.write_manifest(payloads)
        # Write the same again to check it's not updated.
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["Line1", "Extra line"])

        # Act
        new_payloads = patched_publisher.get_upload_payloads()

        # Assert
        assert set(new_payloads) == set(["file01.txt"])  # Nothing to upload!

    def test__get_new_files(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange

        # Write a file and make sure it's in the manifest
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["Line1"])
        payloads = patched_publisher.get_upload_payloads()
        payloads["file01.txt"].pop("data")
        patched_publisher.write_manifest(payloads)
        # Write the same again to check it's not updated.
        with open(web_base_path / "file02.txt", "w+") as f:
            f.writelines(["line in other file"])

        # Act
        new_payloads = patched_publisher.get_upload_payloads()

        # Assert
        assert set(new_payloads) == set(["file02.txt"])


class Test__UploadModifiedFiles:
    def test__upload_modified(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange
        # Write a file and make sure it's in the manifest
        assert not patched_publisher.manifest_filepath.exists()
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["line1"])

        # Act
        success, missing, failed = patched_publisher.upload_modified_files()

        # Assert
        assert set(success) == set(["file01.txt"])

        assert patched_publisher.manifest_filepath.exists()

    def test__no_manifest_record_failed_files(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange
        fpath = web_base_path / "file01.txt"
        with open(fpath, "w+") as f:
            f.writelines(["line1"])
        success_sha = hashlib.sha1(fpath.read_bytes()).hexdigest()
        patched_publisher.upload_modified_files()
        # Write to the same file again, but tell it to fail.
        with open(fpath, "w+") as f:
            f.writelines(["please fail"])
        failed_sha = hashlib.sha1(fpath.read_bytes()).hexdigest()

        # Act
        success, missing, failed = patched_publisher.upload_modified_files()

        # Assert
        assert success_sha != ""  # We actually got something
        assert success_sha != failed_sha

        assert set(success) == set()
        assert set(failed) == set(["file01.txt"])
        new_manifest = patched_publisher.load_manifest()
        assert new_manifest["file01.txt"]["sha"] == success_sha
        assert new_manifest["file01.txt"]["sha"] != failed_sha  # a bit silly


class Test__Deployment:
    def test__deploy(self, patched_publisher: VercelPublisher, web_base_path: Path):
        # Arrange
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["line1"])
        patched_publisher.upload_modified_files()

        # Act
        resp = patched_publisher.deploy()

        # Assert
        resp.status_code == 200  # Not really sure how to test this one.


class Test__Publish:
    def test__test_publish(
        self, patched_publisher: VercelPublisher, web_base_path: Path
    ):
        # Arrange
        with open(web_base_path / "file01.txt", "w+") as f:
            f.writelines(["line1"])

        # Act
        patched_publisher.publish()
