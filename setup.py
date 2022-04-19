import codecs
import os.path

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname("__file__"))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

LONG_DESCRIPTION = """Scikit-identification is a package based on CASADI and scikit-learn."""

DISTNAME = "skidentification"
DESCRIPTION = "skmid: A set of python modules for dynamic model identification"

MAINTAINER = "Giovanni Licitra"
MAINTAINER_EMAIL = "gianni.licitra7@gmail.com"
URL = "https://github.com/G-Licitra/scikit-identification"
DOWNLOAD_URL = "https://github.com/G-Licitra/scikit-identification"
LICENSE = "Apache License 2.0"

install_requires = [
    "numpy>=1.19",
    "scipy>=1.7",
    "pandas>=1.4",
    "matplotlib>=3.5.1",
    # "scikit-identification",
    # "skidentification",
    "seaborn>=0.11",
    "casadi>=3.5",
]

tests_require = [
    "pytest>=6.2.0",
    "pytest-cov>=2.11.0",
    "pytest-mock>=3.5.0",
    "pytest-mpl>=0.12",
]

setup_requires: list = []

packages = find_packages()

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

setup(
    name=DISTNAME,
    version=get_version("skmid/__init__.py"),
    description=DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    classifiers=CLASSIFIERS,
)
