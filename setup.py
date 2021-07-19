
# Copyright (c) Ming Ding, et al. in KEG, Tsinghua University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

from setuptools import find_packages, setup


def _requirements():
    return Path("requirements.txt").read_text()

setup(
    name="cogdata",
    version="0.0.0",
    description="A lightweight data management and preprocessing tool.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=_requirements(),
    entry_points={"console_scripts": ["cogdata = cogdata.cli:cli"]},
    packages=find_packages(),
    url="https://github.com/Sleepychord/cogdata",
    author="Ming Ding, Yuxiang Chen, Wendi Zheng",
    maintainer_email="dm_thu@qq.com",
    scripts={"scripts/install_unrarlib.sh"},
    python_requires=">=3.4"
)