import requests


class LasairClient:
    def __init__(
        self, token, endpoint="https://api.lasair.lsst.ac.uk/api", timeout=60.0
    ):
        self.headers = {"Authorization": f"Token {token}"}
