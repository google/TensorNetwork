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
from typing import Any, Optional, Tuple
from tensornetwork.backends.numpy import numpy_backend
import numpy

Tensor = Any


class JaxBackend(numpy_backend.NumPyBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self, dtype: Optional[numpy.dtype] = None):
    super(JaxBackend, self).__init__()
    try:
      import jax
      jax.config.update("jax_enable_x64", True)
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

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[numpy.dtype] = None,
            seed: Optional[int] = None) -> Tensor:
    if not seed:
      seed = numpy.random.randint(0, 2**63)
    key = self.jax.random.PRNGKey(seed)
    if not dtype:
      dtype = self.dtype
    if not dtype:
      dtype = numpy.float64
    if dtype is self.np.complex128:
      key_2 = self.jax.random.PRNGKey(seed + 1)
      return self.jax.random.normal(
          key, shape,
          dtype=numpy.float64) + numpy.complex128(1j) * self.jax.random.normal(
              key_2, shape, dtype=numpy.float64)
    if dtype is self.np.complex64:
      key_2 = self.jax.random.PRNGKey(seed + 1)
      return self.jax.random.normal(
          key, shape,
          dtype=numpy.float32) + numpy.complex64(1j) * self.jax.random.normal(
              key_2, shape, dtype=numpy.float32)

    return self.jax.random.normal(key, shape).astype(dtype)
