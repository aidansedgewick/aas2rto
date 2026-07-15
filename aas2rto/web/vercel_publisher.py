import hashlib
import json
import requests
import tqdm
from logging import getLogger
from pathlib import Path

from aas2rto.utils import QueryTracker, check_unexpected_config_keys


def _iter_files(site_base: Path):
    for path in site_base.rglob("*"):
        if path.is_file():
            yield path


class VercelPublisher:
    """Does not rely on npm vercel package to avoid dependencies."""

    vercel_api = "https://api.vercel.com"

    default_config = {
        "project_token": None,
        "project_name": None,
        "deployment": "production",
        "manifest_filename": "vercel_manifest",
        "max_failed_uploads": 25,
    }

    def __init__(self, config: dict, site_base_path: Path, manifest_parent: Path):
        self.config = self.default_config.copy()
        self.config.update(config)
        check_unexpected_config_keys(
            self.config,
            self.default_config,
            name="web.static_pages.vercel",
            raise_exc=True,
        )

        if self.config["project_token"] is None:
            raise ValueError(f"vercel publisher config missing 'project_token'")
        if self.config["project_name"] is None:
            raise ValueError(f"vercel publisher config missing 'project_name'")

        self.site_base_path = Path(site_base_path)

        manifest_filename = self.config["manifest_filename"]
        self.manifest_filepath = manifest_parent / f"{manifest_filename}.json"

        self.logger = getLogger("vercel_publisher")

    def load_manifest(self) -> dict:
        if self.manifest_filepath.exists():
            with open(self.manifest_filepath, "r") as f:
                return json.load(f)
        return {}

    def publish(self):
        vercel_manifest = self.load_manifest()

        skipped = 0
        file_deploy_payloads: list[dict] = []
        upload_payloads: list[dict] = []
        for path in _iter_files(self.site_base_path):
            file_data = path.read_bytes()
            digest = hashlib.sha1(file_data).hexdigest()
            rel_filepath = path.relative_to(self.site_base_path).as_posix()

            # Track ALL files for deployment...
            payload = {"file": rel_filepath, "sha": digest, "size": len(file_data)}
            file_deploy_payloads.append(payload)

            existing_digest = vercel_manifest.get(rel_filepath, "")
            if digest == existing_digest:
                skipped = skipped + 1
                continue

            # ...but only upload ones which are updated.
            payload = {**payload, "data": file_data}
            upload_payloads.append(payload)

            # Update manifest digets AFTER successful upload

        token = self.config["project_token"]
        session_headers = {"Authorization": f"Bearer {token}"}

        ##== Upload files and pages
        self.logger.info("start")
        qtracker = QueryTracker.start(
            max_failed_queries=self.config["max_failed_uploads"]
        )
        self.logger.info(f"upload {len(upload_payloads)} files")
        with requests.Session() as session:
            session.headers.update(session_headers)
            for payload in tqdm.tqdm(upload_payloads):
                rel_filepath = payload["file"]
                data = payload["data"]
                upload_headers = {
                    "x-vercel-digest": payload["sha"],
                    "Content-Length": str(len(data)),
                }
                response: requests.Response = session.post(
                    f"{self.vercel_api}/v2/files",
                    params={},
                    headers=upload_headers,
                    data=data,
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    qtracker.track_failure(rel_filepath)
                    continue

                vercel_manifest[rel_filepath] = payload["sha"]
                qtracker.track_success(rel_filepath)

        qtracker.log_summary(name="vercel publish")

        ##== Request deployment
        self.logger.info("request deployment")
        deployment_payload = {
            "name": self.config["project_name"],
            "project": self.config["project_name"],
            "target": self.config["deployment"],
            "files": file_deploy_payloads,
        }
        deployment_headers = {**session_headers, "Content-Type": "application/json"}
        response: requests.Response = requests.post(
            f"{self.vercel_api}/v13/deployments",
            params={},
            headers=deployment_headers,
            json=deployment_payload,
        )

        try:
            response.raise_for_status()
            self.logger.info(f"deployment status {response.status_code}")
        except requests.HTTPError as e:
            msg = (
                f"deployment failed: status {response.status_code}\n"
                f"    {e}\n    {response.reason}"
            )
            self.logger.error(msg)
