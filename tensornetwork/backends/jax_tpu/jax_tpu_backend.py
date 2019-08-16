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
from typing import Optional, Any, Sequence, Tuple
from tensornetwork.backends import base_backend
from tensornetwork.backends.shell import shell_backend
import jax
import numpy as onp
import logging

Tensor = Any


def optimized_shape(shape):
  size = 1
  # This is disabled since the return below will
  # always run since we have a `or i == 0`.
  #pylint: disable=inconsistent-return-statements
  for i, dim in reversed(list(enumerate(shape))):
    size *= dim
    if size >= 128 or i == 0:
      first_expensive_axis = i
      new_shape = (int(onp.prod(shape[:i])),) + (size,)
      return new_shape, first_expensive_axis


class JaxTPUTensor():

  def __init__(self, concrete_tensor, virtual_tensor):
    assert isinstance(virtual_tensor, shell_backend.ShellTensor)
    concrete_tensor = self.to_device_array(concrete_tensor)
    if onp.prod(concrete_tensor.shape) != onp.prod(virtual_tensor.shape):
      raise ValueError("Virtual shapes and concrete shapes do not math."
                       "Virtual shape: {}, concrete shape: {}".format(
                           virtual_tensor.shape, concrete_tensor.shape))
    self.virtual_tensor = virtual_tensor
    self.set_optimized_concrete_tensor(concrete_tensor)

  @property
  def shape(self):
    return self.virtual_tensor.shape

  @property
  def real_tensor(self):
    return jax.numpy.reshape(self.concrete_tensor, self.virtual_tensor.shape)

  def set_optimized_concrete_tensor(self, tensor):
    shape, self.first_expensive_axis = optimized_shape(tensor.shape)
    if shape != tensor.shape:
      self.concrete_tensor = jax.numpy.reshape(tensor, shape)
    else:
      self.concrete_tensor = tensor

  def to_device_array(self, tensor):
    return jax.jit(lambda x: x)(tensor)

# pylint: disable=abstract-method
class JaxTPUBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self):
    super(JaxTPUBackend, self).__init__()
    self.shell_backend = shell_backend.ShellBackend()

  def concat(self, values: Tensor, axis: int) -> Tensor:
    return onp.concatenate(values, axis)

  def shape(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def prod(self, values: Tensor) -> Tensor:
    return onp.prod(values)

  def reshape(self, tensor, shape):
    tmp_tensor = jax.numpy.reshape(tensor.concrete_tensor, shape)
    return JaxTPUTensor(tmp_tensor, shell_backend.ShellTensor(tuple(shape)))

  def transpose(self, tensor, perm):
    tmp_tensor = jax.numpy.reshape(tensor.concrete_tensor, tensor.shape)
    tmp_tensor = jax.numpy.transpose(tmp_tensor, perm)
    return JaxTPUTensor(tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))

  def convert_to_tensor(self, tensor):
    if isinstance(tensor, JaxTPUTensor):
      return tensor
    return JaxTPUTensor(tensor, shell_backend.ShellTensor(tensor.shape))

  def tensordot(self, a, b, axes):
    use_optimized = True
    for contraction_axes, tensor in zip(axes, [a, b]):
      for axis in contraction_axes:
        if axis >= tensor.first_expensive_axis:
          use_optimized = False
    if use_optimized:
      resulting_tensor = optimized_dot_general(a, b, axes)
    else:
      # Revert to normal tensordot.
      concrete_a = jax.numpy.reshape(a.concrete_tensor, a.shape)
      concrete_b = jax.numpy.reshape(b.concrete_tensor, b.shape)
      resulting_tensor = jax.numpy.tensordot(concrete_a, concrete_b, axes)
    new_virtual = self.shell_backend.tensordot(a.virtual_tensor,
                                               b.virtual_tensor, axes)
    return JaxTPUTensor(resulting_tensor, new_virtual)


def optimized_dot_general(a, b, contracting_axes):
  new_tensors = []
  for i, tensor in enumerate([a, b]):
    # These reshapes are much cheaper than the reshapes done in tensordot.
    axes = contracting_axes[i]
    new_dim = onp.prod([tensor.shape[index] for index in axes])
    new_shape = (
        list(tensor.shape[:tensor.first_expensive_axis]) +
        [tensor.concrete_tensor.shape[-1]])
    tmp_tensor = jax.numpy.reshape(tensor.concrete_tensor, new_shape)
    perm = axes + sorted(list(set(range(len(tmp_tensor.shape))) - set(axes)))
    tmp_tensor = jax.numpy.transpose(tmp_tensor, perm)
    tmp_tensor = jax.numpy.reshape(tmp_tensor, [new_dim] +
                                   list(tmp_tensor.shape[len(axes):]))
    new_tensors.append(tmp_tensor)

  return jax.lax.dot_general(new_tensors[0], new_tensors[1],
                             (((0,), (0,)), (tuple(), tuple())))
