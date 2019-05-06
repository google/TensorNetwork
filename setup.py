# Copyright 2019 The TensorNetwork Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from setuptools import find_packages, setup

# This reads the __version__ variable from cirq/_version.py
__version__ = ''
description = ('A high level tensor network API for tensorflow.')

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='tensornetwork',
    version='0.0.1',
    url='http://github.com/google/TensorNetwork',
    author='The TensorNetwork Developers',
    author_email='chaseriley@googleg.com',
    python_requires=('>=3.5.0'),
    install_requires=requirements,
    license='Apache 2',
    description=description,
    packages=['tensornetwork'],
)
