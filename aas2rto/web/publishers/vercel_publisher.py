import datetime
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
        "size": len(file_data),
    }
    if with_data:
        payload["data"] = file_data
    return payload

class VercelProjectVerificationError(requests.HTTPError):
    pass

class VercelClient:

    vercel_api = "https://api.vercel.com"

    def __init__(
        self, project_token: str, project_name: str, deployment: str = "production"
    ):
        self.project_token = project_token
        self.project_name = project_name
        self.deployment = deployment
        self.logger = getLogger("vercel_client")
        self.default_headers = {
            "Authorization": f"Bearer {self.project_token}"
        }

    def verify_project_exists(self):
        headers = {
            "Authorization": f"Bearer {self.project_token}",
            "Content-Type": "application/json",
        }
        url = f"{self.vercel_api}/v9/projects/{self.project_name}"
        with requests.Session() as session:
            session.headers.update(headers)
            response: requests.Response = session.get(url, headers=headers)

        try:
            response.raise_for_status()
            self.logger.info(f"project '{self.project_name}' verified")
        except requests.HTTPError as e:
            logger.error(e)
            msg = (
                f"project '{self.project_name}' verification failed!\n"
                f"    \033[31;1mstatus {response.status_code}: {response.reason}\033[0m\n"
                f"        - check the project name, or\n"
                f"        - create the project at vercel.com/drop, or\n"
                f"        - switch web.static_pages.vercel.use = False"
            )
            self.logger.error(msg)
            raise VercelProjectVerificationError(msg)

    def upload_single_file(
        self,
        payload: dict[str, str],
        session: requests.Session = None,
        attempt: int = 0,
        max_attempts: int = 3,
    ):
        api_url = f"{self.vercel_api}/v2/files"
        is_local_session = session is None
        if session is None:
            session = requests.Session()
            session_headers = {"Authorization": f"Bearer {self.project_token}"}
            session.headers.update(session_headers)
            close_session = True # Make sure session is close if not passed as ctx mgr.

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
        if is_local_session:
            session.close()

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
                return self.upload_single_file(payload, session=session, **attempt_kws)
            else:
                raise
        return response

    def deploy(
        self,
        files_array: list[dict],
    ):
        ##== Request deployment
        self.logger.info("request deployment")
        deployment_payload = {
            "name": self.project_name,
            "target": self.deployment,
            "files": files_array,
        }
        deployment_headers = {
            "Authorization": f"Bearer {self.project_token}",
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

    def get_deployment_history(self, limit: int=100):
        self.logger.info("get vercel deployments")

        url = f"{self.vercel_api}/v7/deployments"

        params = {"limit": limit, "name": self.project_name}

        more_deployments_available = True
        until = None
        deployment_records = []

        with requests.Session() as session:
            session.headers.update(self.default_headers)
            while more_deployments_available:
                if until is not None:
                    params["until"] = until # pagination param, request next page

                response: requests.Response = session.get(
                    url, headers=self.default_headers, params=params
                )

                response.raise_for_status()

                data = response.json()
                deployments = data["deployments"]

                self.logger.debug(f"Found {len(deployments)}/{limit} deployments")
                deployment_records.extend(deployments)

                until = data["pagination"]["next"]
                if until is None:
                    more_deployments_available = False

        self.logger.info(f"Got {len(deployment_records)} deployment records total")
        return deployment_records

    def delete_deployment(self, deployment_id: int, session: requests.Session = None):
        url = f"{self.vercel_api}/v13/deployments/{deployment_id}"

        logger.info(f"about to del at {url}")
        is_local_session = session is None # Close sess later if not passed as ctx mgr
        if session is None:
            session = requests.Session()
            session.headers.update(self.default_headers)
            close_session = True 

        response: requests.Response = session.delete(url)
        if is_local_session:
            session.close()

        logger.info(f"get response {response} {response.reason}")

        response.raise_for_status()


class VercelPublisher:
    """Does not rely on npm vercel package to avoid dependencies."""

    default_config = {
        "project_token": None,
        "project_name": None,
        "use_direct_deployment": False,
        "deployment": "production",
        "manifest_filestem": "vercel_manifest",
        "verify_project": True,
        "max_failed_uploads": 5,
        "max_upload_attempts": 3,
        "publish_interval": 1800.0,
        "max_deployments": 3,
        "max_failed_cleanups": 5,
        "cleanup_interval": 3600.0,
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

        self.vercel_client = VercelClient(
            self.config["project_token"],
            self.config["project_name"],
            deployment=self.config["deployment"],
        )
        if self.config["verify_project"]:
            self.vercel_client.verify_project_exists()

        self.last_publish_time = 0.0
        self.last_deployment_cleanup = 0.0

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

        self.upload_modified_files()  # Better to upload few files each loop.

        publish_interval = self.config["publish_interval"]
        elapsed = time.perf_counter() - self.last_publish_time
        if elapsed < publish_interval:
            msg = f"skip vercel publish: {elapsed:.1f}s<{publish_interval:.1f} required"
            logger.info(msg)
            return

        self.deploy()
        self.last_publish_time = time.perf_counter()

    def deploy(self):

        manifest = self.load_manifest()

        files_array = []
        for rel_path, payload in manifest.items():
            clean_payload = payload.copy()
            clean_payload["size"] = int(payload["size"])
            files_array.append(clean_payload)

        # Actual deployment
        return self.vercel_client.deploy(files_array)
    

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
                if not qtracker.safe_to_query():
                    break

                try:
                    self.vercel_client.upload_single_file(
                        payload,
                        session=session,
                        max_attempts=self.config["max_upload_attempts"],
                    )
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

    def cleanup_old_deployments(self, force: bool=False):
        self.logger.info("start deployment clean-up")

        cleanup_interval = self.config["cleanup_interval"]
        elapsed = time.perf_counter() - self.last_deployment_cleanup
        
        if elapsed < cleanup_interval and not force:
            msg = f"skip clean-up: {elapsed:.1f}s (<{cleanup_interval} since last"
            logger.info(msg)
            return
        
        deployment_records = self.vercel_client.get_deployment_history()
        n_deployments = self.config["max_deployments"]

        deployment_records.sort(key=lambda dpl: dpl["created"]) 
        deployments_to_clear = deployment_records[:-n_deployments]

        qtracker = QueryTracker.start(
            max_failed_queries=self.config["max_failed_cleanups"], max_query_time=60.0
        )
        self.logger.info(f"attempt to clear {len(deployments_to_clear)} deployments")
        for deployment in deployments_to_clear:
            if not qtracker.safe_to_query():
                break

            # Loop in reverse: if starts to fail, we'd prefer to delete the oldest.
            deployment_id = deployment["uid"]
            created = deployment["created"]
            seconds = created / 1000.0
            iso = datetime.datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S")
            msg = f"del deployment {iso}\n    (id {deployment_id})"
            self.logger.info(msg)
            try:
                self.vercel_client.delete_deployment(deployment_id)
                qtracker.track_success(deployment_id)
                logger.info(f"tracking success {deployment_id}")
            except requests.HTTPError as e:
                qtracker.track_failed(deployment_id)

        qtracker.log_summary(
            name="vercel deployment clean-up", 
            tasks_str="requests", 
            data_str="deployments",
        )
        self.last_deployment_cleanup = time.perf_counter()
        return qtracker.success, qtracker.failed

    def perform_all_publisher_tasks(self):
        if self.config["use_direct_deployment"]:
            self.publish()

        self.cleanup_old_deployments()