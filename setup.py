#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="aas2rto",
    version="0.1.0",
    description="aas2rto",
    author="Aidan S",
    author_email="aidan.sedgewick@gmail.com",
    # url='',
    packages=find_packages(),
)

from aas2rto.utils import init_sfd_dustmaps

init_sfd_dustmaps()
