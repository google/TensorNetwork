# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of TensorNetwork structure."""

from typing import Optional, Text, Type
import numpy as np


class TensorNetwork:
  """Implementation of a TensorNetwork."""

  #pylint: disable=unused-argument
  def __init__(self, *args, **kwargs):
    link = ("https://medium.com/@keeper6928/" +
            "upgrading-your-tensornetwork-code-b032f0ab3dd4")
    raise DeprecationWarning("The TensorNetwork object has been DEPRECATED "
                             "and has been removed.\n"
                             "Updating your code is fairly easy. For a "
                             f"detailed tutorial, please visit {link}")
