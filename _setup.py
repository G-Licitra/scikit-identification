#! /usr/bin/env python
#
# Copyright (C) 2022 Giovanni Licitra
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


LONG_DESCRIPTION = """Scikit-identification is a package based on CASADI and scikit-learn."""

DISTNAME = "skidentification"
DESCRIPTION = "skmid: A set of python modules for dynamic model identification"

MAINTAINER = "Giovanni Licitra"
MAINTAINER_EMAIL = "gianni.licitra7@gmail.com"
URL = "https://github.com/G-Licitra/scikit-identification"
DOWNLOAD_URL = "https://github.com/G-Licitra/scikit-identification"
VERSION = "0.1.0"
LICENSE = "Apache License 2.0"
# PACKAGE_DATA = {'skmid.data.icons': ['*.svg']}
# PACKAGES=['skmid']
# PACKAGE_DIR={'skidentification': 'scikit-identification/skmid'}

INSTALL_REQUIRES = [
    "numpy>=1.19",
    "scipy>=1.7",
    "pandas>=1.4" "matplotlib>=3.5.1",
    # "scikit-identification",
    "skidentification",
    "seaborn>=0.11",
    "casadi>=3.5",
]


CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

try:
    from setuptools import setup

    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
                #  packages=PACKAGES,
        #          package_data=PACKAGE_DATA,
        # package_dir=PACKAGE_DIR,
        classifiers=CLASSIFIERS,
    )
