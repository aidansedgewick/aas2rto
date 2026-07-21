import hashlib
import json
import requests
import time
import tqdm
from logging import getLogger
from pathlib import Path

from aas2rto.exc import MissingKeysError
from aas2rto.utils import QueryTracker, check_unexpected_config_keys

logger = getLogger(__name__.split(".")[-1])


def _iter_files(site_base: Path):
    for path in site_base.rglob("*"):
        if path.is_file():
            yield path


def _get_file_payload(filepath: Path, rel_path: Path, with_data: bool = True):

    file_data = Path(filepath).read_bytes()
    digest = hashlib.sha1(file_data).hexdigest()
    payload = {
        "file": str(rel_path),
        "sha": digest,
        "size": str(len(file_data)),
    }
    if with_data:
        payload["data"] = file_data
    return payload


def _upload_single_file(
    payload: dict[str, str],
    api_url=None,
    session: requests.Session = None,
    token: str = None,
    attempt: int = 0,
    max_attempts: int = 3,
):
    api_url = api_url or f"{VercelPublisher.vercel_api}/v2/files"
    if session is None:
        session = requests.Session()
        if token is None:
            msg = "Provide prepared 'requests.Session' object, or 'token' as str"
            raise ValueError()
        session_headers = {"Authorization": f"Bearer {token}"}
        session.headers.update(session_headers)

    rel_path = payload["file"]
    data = payload["data"]
    upload_headers = {
        "x-Vercel-Digest": payload["sha"],
        "Content-Length": str(len(data)),
    }
    response: requests.Response = session.post(
        f"{api_url}",
        params={},
        headers=upload_headers,
        data=data,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        if attempt < max_attempts:
            msg = (
                f"upload {rel_path}:\n"
                f"    failed on attempt {attempt+1}/{max_attempts}"
            )
            logger.error(msg)
            attempt_kws = dict(attempt=attempt + 1, max_attempts=max_attempts)
            return _upload_single_file(
                payload, api_url=api_url, session=session, token=token, **attempt_kws
            )
    return response


class VercelPublisher:
    """Does not rely on npm vercel package to avoid dependencies."""

    vercel_api = "https://api.vercel.com"

    default_config = {
        "project_token": None,
        "project_name": None,
        "deployment": "production",
        "manifest_filestem": "vercel_manifest",
        "max_failed_uploads": 25,
        "publish_interval": 1800.0,
    }

    def __init__(self, config: dict, web_base_path: Path):
        self.config = self.default_config.copy()
        self.config.update(config)
        check_unexpected_config_keys(
            self.config,
            self.default_config,
            name="web.static_pages.vercel",
            raise_exc=True,
        )

        if self.config["project_token"] is None:
            raise MissingKeysError(f"config web.vercel missing 'project_token'")
        if self.config["project_name"] is None:
            raise MissingKeysError(f"config web.vercel missing 'project_name'")

        self.web_base_path = Path(web_base_path)

        manifest_filename = self.config["manifest_filestem"]
        self.manifest_filepath = self.web_base_path.parent / f"{manifest_filename}.json"
        self.last_publish_time = 0.0

        self.logger = getLogger("vercel_publisher")

    def load_manifest(self) -> dict[str, dict[str, str]]:
        if self.manifest_filepath.exists():
            with open(self.manifest_filepath, "r") as f:
                return json.load(f)
        return {}

    def write_manifest(self, manifest: dict[str, dict[str, str]]) -> None:
        with open(self.manifest_filepath, "w+") as f:
            json.dump(manifest, f)
        return

    def publish(self):

        publish_interval = self.config["publish_interval"]
        elapsed = time.perf_counter() - self.last_publish_time
        if elapsed < publish_interval:
            msg = f"skip vercel publish: {elapsed:.1f}s<{publish_interval:.1f} required"
            logger.info(msg)
            return

        self.upload_modified_files()
        self.deploy()

    def get_upload_payloads(self, with_data: bool = True):
        vercel_manifest = self.load_manifest()

        skipped = []
        upload_payloads: dict[str, dict[str, str]] = {}
        for filepath in _iter_files(self.web_base_path):
            rel_path = str(filepath.relative_to(self.web_base_path))  # dict key is str
            payload = _get_file_payload(filepath, rel_path, with_data=with_data)

            existing_sha = vercel_manifest.get(rel_path, {}).get("sha", "")
            if payload["sha"] == existing_sha:
                skipped.append(rel_path)
                continue
            upload_payloads[str(rel_path)] = payload
        self.logger.info(f"skip {len(skipped)} unmodified files")
        return upload_payloads

    def upload_modified_files(self):

        upload_payloads = self.get_upload_payloads()  #

        ##== Upload files and pages
        token = self.config["project_token"]
        session_headers = {"Authorization": f"Bearer {token}"}

        qtracker = QueryTracker.start(
            max_failed_queries=self.config["max_failed_uploads"]
        )
        self.logger.info(f"vercel upload {len(upload_payloads)} files")
        manifest_updates = {}
        with requests.Session() as session:
            session.headers.update(session_headers)
            for rel_path, payload in tqdm.tqdm(upload_payloads.items()):
                response = _upload_single_file(
                    payload, api_url=f"{self.vercel_api}/v2/files", session=session
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    qtracker.track_failed(rel_path)
                    continue
                qtracker.track_success(rel_path)
                payload.pop("data")  # don't want to dump data the manifest...
                manifest_updates[rel_path] = payload

        qtracker.log_summary(name="vercel upload")
        self.logger.info("write updated manifest")
        manifest = self.load_manifest()
        manifest.update(manifest_updates)
        self.write_manifest(manifest)
        return qtracker.success, qtracker.missing, qtracker.failed

    def deploy(self):

        ##== Decide files
        manifest = self.load_manifest()
        files_array = [payload for rel_path, payload in manifest.items()]

        ##== Request deployment
        self.logger.info("request deployment")
        deployment_payload = {
            "name": self.config["project_name"],
            "target": self.config["deployment"],
            "files": files_array,
        }
        token = self.config["project_token"]
        deployment_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        with requests.Session() as session:
            session.headers.update(deployment_headers)
            response: requests.Response = session.post(
                f"{self.vercel_api}/v13/deployments", params={}, json=deployment_payload
            )

        ##== Check status
        self.logger.info(f"deployment status {response.status_code}")
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            msg = (
                f"deployment failed: status {response.status_code}\n"
                f"    {e}\n    {response.reason}"
            )
            self.logger.error(msg)
        return response
