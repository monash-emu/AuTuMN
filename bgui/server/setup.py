#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from setuptools import setup, find_packages

with open("./_version.py", "r") as f:
    version_file = {}
    exec(f.read(), version_file)
    version = version_file["__version__"]

CLASSIFIERS = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GPLv3',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Development Status :: 4 - Beta',
    'Programming Language :: Python :: 2.7',
]

setup(
    name='Base GUI',
    version=version,
    author='Bosco Ho',
    author_email='apposite@gmail.com',
    description='Webserver for python simulations"',
    long_description=open('README.md').read(),
    url='http://github.com/boscoh/basegui',
    platforms=['OS Independent'],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
)
