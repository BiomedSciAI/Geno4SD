#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_namespace_packages


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(os.path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of requirements for core packages
geno4sd_requirements = []
with open(os.path.join(HERE, "requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            geno4sd_requirements.append(line.strip())

from geno4sd.version import __version__

version = __version__

setup(
    name="geno4sd",
    version=version,
    description="An open source python data toolkit for the analysis across omics scales",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiomedSciAI/Geno4SD/",
    author="IBM Research - Computational Genomics",
    author_email="futro@us.ibm.com",
    packages=find_namespace_packages(),
    license="Apache License 2.0",
    install_requires=geno4sd_requirements,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)