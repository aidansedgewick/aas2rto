#!/usr/bin/env python

from distutils.core import setup

setup(
    name="dk154_targets",
    version="0.1.0",
    description="dk154_targets",
    author="Aidan S",
    author_email="aidan.sedgewick@gmail.com",
    # url='',
    packages=[
        "dk154_targets",
    ],
)

from dk154_targets.paths import build_paths

#build_paths()
