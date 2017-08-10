#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from setuptools import setup, find_packages

import re
regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(regex, open("_version.py").read(), re.M)
if mo:
    version = mo.group(1)

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
