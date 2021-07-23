
# Copyright (c) Ming Ding, et al. in KEG, Tsinghua University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

from setuptools import find_packages, setup
from cogdata.version import __version__

def _requirements():
    return Path("requirements.txt").read_text()

setup(
    name="cogdata",
    version=__version__,
    description="A lightweight data management and preprocessing tool.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=_requirements(),
    entry_points={"console_scripts": ["cogdata = cogdata.cli:main"]},
    packages=find_packages(),
    url="https://github.com/Sleepychord/cogdata",
    author="Ming Ding, Yuxiang Chen, Wendi Zheng",
    author_email="dm_thu@qq.com",
    scripts={"scripts/install_unrarlib.sh"},
    include_package_data=True,
    python_requires=">=3.5",
    license="MIT license"
)