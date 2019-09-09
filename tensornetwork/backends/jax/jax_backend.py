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
from typing import Any, Optional
from tensornetwork.backends.numpy import numpy_backend
import numpy


Tensor = Any

supported_dtypes = numpy_backend.supported_dtypes


class JaxBackend(numpy_backend.NumPyBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self, dtype: Optional[numpy.dtype] = None):
    super(JaxBackend, self).__init__()
    try:
      import jax
    except ImportError:
      raise ImportError("Jax not installed, please switch to a different "
                        "backend or install Jax.")
    self.jax = jax
    self.np = self.jax.numpy
    self.name = "jax"
    self.dtype = dtype

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    result = self.jax.jit(lambda x: x)(tensor)
    if self.dtype is not None and result.dtype != self.dtype:
      raise TypeError(
          "Backend '{}' cannot convert tensor of dtype {} to dtype {}".format(
              self.name, result.dtype, self.dtype))
    return result

  def concat(self, values: Tensor, axis: int) -> Tensor:
    return numpy.concatenate(values, axis)

  #TODO: add initializers (randn, zeros, eye, ones)
