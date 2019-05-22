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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any
from tensornetwork.backends.numpy import numpy_backend
import numpy
Tensor = Any


class JaxBackend(numpy_backend.NumPyBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(JaxBackend, self).__init__()
    try:
      import jax
    except ImportError:
      raise AssertionError("jax is not installed.")
    self.jax = jax
    self.np = jax.numpy
    self.name = "jax"

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    return self.jax.jit(lambda x: x)(tensor)

  def concat(self, values: Tensor, axis: int) -> Tensor:
    return numpy.concatenate(values, axis)
