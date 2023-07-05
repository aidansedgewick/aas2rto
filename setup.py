#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="dk154_targets",
    version="0.1.0",
    description="dk154_targets",
    author="Aidan S",
    author_email="aidan.sedgewick@gmail.com",
    # url='',
    packages=find_packages(),
)

from dk154_targets.paths import build_paths
from dk154_targets.utils import init_sfd_dustmaps

build_paths()
init_sfd_dustmaps()
