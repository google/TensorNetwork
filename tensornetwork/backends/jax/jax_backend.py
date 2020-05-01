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

from typing import Any, Optional, Tuple, Callable, List, Text, Type
from tensornetwork.backends.numpy import numpy_backend
import numpy as np
from tensornetwork.backends.jax import jitted_functions
from functools import partial

Tensor = Any


class JaxBackend(numpy_backend.NumPyBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self, dtype: Optional[np.dtype] = None):
    super(JaxBackend, self).__init__()
    try:
      #pylint: disable=import-outside-toplevel
      import jax
    except ImportError:
      raise ImportError("Jax not installed, please switch to a different "
                        "backend or install Jax.")
    self.jax = jax
    self.np = self.jax.numpy
    self.sp = self.jax.scipy
    self.name = "jax"
    self._dtype = np.dtype(dtype) if dtype is not None else None

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    return self.np.asarray(tensor)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    return self.jax.lax.dynamic_slice(tensor, start_indices, slice_sizes)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None,
            seed: Optional[int] = None) -> Tensor:
    if not seed:
      seed = np.random.randint(0, 2**63)
    key = self.jax.random.PRNGKey(seed)

    dtype = dtype if dtype is not None else np.dtype(np.float64)

    def cmplx_randn(complex_dtype, real_dtype):
      real_dtype = np.dtype(real_dtype)
      complex_dtype = np.dtype(complex_dtype)

      key_2 = self.jax.random.PRNGKey(seed + 1)

      real_part = self.jax.random.normal(key, shape, dtype=real_dtype)
      complex_part = self.jax.random.normal(key_2, shape, dtype=real_dtype)
      unit = (
          np.complex64(1j)
          if complex_dtype == np.dtype(np.complex64) else np.complex128(1j))
      return real_part + unit * complex_part

    if np.dtype(dtype) is np.dtype(self.np.complex128):
      return cmplx_randn(dtype, self.np.float64)
    if np.dtype(dtype) is np.dtype(self.np.complex64):
      return cmplx_randn(dtype, self.np.float32)

    return self.jax.random.normal(key, shape).astype(dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[np.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:
    if not seed:
      seed = np.random.randint(0, 2**63)
    key = self.jax.random.PRNGKey(seed)

    dtype = dtype if dtype is not None else np.dtype(np.float64)

    def cmplx_random_uniform(complex_dtype, real_dtype):
      real_dtype = np.dtype(real_dtype)
      complex_dtype = np.dtype(complex_dtype)

      key_2 = self.jax.random.PRNGKey(seed + 1)

      real_part = self.jax.random.uniform(
          key,
          shape,
          dtype=real_dtype,
          minval=boundaries[0],
          maxval=boundaries[1])
      complex_part = self.jax.random.uniform(
          key_2,
          shape,
          dtype=real_dtype,
          minval=boundaries[0],
          maxval=boundaries[1])
      unit = (
          np.complex64(1j)
          if complex_dtype == np.dtype(np.complex64) else np.complex128(1j))
      return real_part + unit * complex_part

    if np.dtype(dtype) is np.dtype(self.np.complex128):
      return cmplx_random_uniform(dtype, self.np.float64)
    if np.dtype(dtype) is np.dtype(self.np.complex64):
      return cmplx_random_uniform(dtype, self.np.float32)

    return self.jax.random.uniform(
        key, shape, minval=boundaries[0], maxval=boundaries[1]).astype(dtype)

  def eigs(self,
           A: Callable,
           initial_state: Optional[Tensor] = None,
           num_krylov_vecs: Optional[int] = 200,
           numeig: Optional[int] = 1,
           tol: Optional[float] = 1E-8,
           which: Optional[Text] = 'LR',
           maxiter: Optional[int] = None,
           dtype: Optional[Type] = None) -> Tuple[List, List]:
    raise NotImplementedError("Backend '{}' has not implemented eigs.".format(
        self.name))

  def eigsh_lanczos(
      self,
      A: Callable,
      args: List[Tensor],
      initial_state: Optional[Tensor] = None,
      num_krylov_vecs: Optional[int] = 200,
      numeig: Optional[int] = 1,
      tol: Optional[float] = 1E-8,
      delta: Optional[float] = 1E-8,
      ndiag: Optional[int] = 20,
      reorthogonalize: Optional[bool] = False) -> Tuple[List, List]:

    if num_krylov_vecs < numeig:
      raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if numeig > 1 and not reorthogonalize:
      raise ValueError(
          "Got numeig = {} > 1 and `reorthogonalize = False`. "
          "Use `reorthogonalize=True` for `numeig > 1`".format(numeig))

    if (initial_state is not None) and hasattr(A, 'shape'):
      if initial_state.shape != A.shape[1]:
        raise ValueError(
            "A.shape[1]={} and initial_state.shape={} are incompatible.".format(
                A.shape[1], initial_state.shape))

    if initial_state is None:
      if not hasattr(A, 'shape'):
        raise AttributeError("`A` has no  attribute `shape`. Cannot initialize "
                             "lanczos. Please provide a valid `initial_state`")
      if not hasattr(A, 'dtype'):
        raise AttributeError(
            "`A` has no  attribute `dtype`. Cannot initialize "
            "lanczos. Please provide a valid `initial_state` with "
            "a `dtype` attribute")

      initial_state = self.randn(A.shape[1], A.dtype)
    if not isinstance(initial_state, self.np.ndarray):
      raise TypeError("Expected a `jax.array`. Got {}".format(
          type(initial_state)))

    if not hasattr(self, '_jaxlan'):
      #avoid retracing
      self._jaxlan = jitted_functions._generate_jitted_eigsh_lanczos(self.jax)

    return self._jaxlan(A, args, initial_state, num_krylov_vecs, numeig, delta,
                        reorthogonalize)

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    return self.jax.ops.index_update(tensor, mask, assignee)
