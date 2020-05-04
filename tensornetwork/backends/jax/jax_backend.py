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

from typing import Any, Optional, Tuple, Callable, List, Text, Type, Sequence
from tensornetwork.backends import base_backend
from tensornetwork.backends.numpy import decompositions
import numpy as np
from tensornetwork.backends.jax import jitted_functions
from functools import partial

Tensor = Any

# pylint: disable=abstract-method

_CACHED_MATVECS = {}


class JaxBackend(base_backend.BaseBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self, dtype: Optional[np.dtype] = None):
    # pylint: disable=global-variable-undefined
    global libjax  # Jax module
    global jnp  # jax.numpy module
    global jsp  # jax.scipy module
    super(JaxBackend, self).__init__()
    try:
      #pylint: disable=import-outside-toplevel
      import jax
    except ImportError:
      raise ImportError("Jax not installed, please switch to a different "
                        "backend or install Jax.")
    libjax = jax
    jnp = libjax.numpy
    jsp = libjax.scipy
    self.name = "jax"
    self._dtype = np.dtype(dtype) if dtype is not None else None

  def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[Sequence[int]]):
    return jnp.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor):
    return jnp.reshape(tensor, np.asarray(shape).astype(np.int32))

  def transpose(self, tensor, perm):
    return jnp.transpose(tensor, perm)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    return libjax.lax.dynamic_slice(tensor, start_indices, slice_sizes)

  def svd_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd_decomposition(
        jnp,
        tensor,
        split_axis,
        max_singular_values,
        max_truncation_error,
        relative=relative)

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr_decomposition(jnp, tensor, split_axis)

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq_decomposition(jnp, tensor, split_axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> Tensor:
    return np.prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return jnp.sqrt(tensor)

  def diag(self, tensor: Tensor) -> Tensor:
    if len(tensor.shape) != 1:
      raise TypeError("Only one dimensional tensors are allowed as input")
    return jnp.diag(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if (not isinstance(tensor, jnp.ndarray) and not jnp.isscalar(tensor)):
      raise TypeError("Expected a `jnp.array` or scalar. Got {}".format(
          type(tensor)))
    result = jnp.asarray(tensor)
    return result

  def trace(self, tensor: Tensor) -> Tensor:
    # Default np.trace uses first two axes.
    return jnp.trace(tensor, axis1=-2, axis2=-1)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return jnp.tensordot(tensor1, tensor2, 0)

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    return jnp.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return jnp.linalg.norm(tensor)

  def eye(self,
          N,
          dtype: Optional[np.dtype] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else jnp.float64
    return jnp.eye(N, M=M, dtype=dtype)

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[np.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else jnp.float64
    return jnp.ones(shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else jnp.float64
    return jnp.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None,
            seed: Optional[int] = None) -> Tensor:
    if not seed:
      seed = np.random.randint(0, 2**63)
    key = libjax.random.PRNGKey(seed)

    dtype = dtype if dtype is not None else np.dtype(np.float64)

    def cmplx_randn(complex_dtype, real_dtype):
      real_dtype = np.dtype(real_dtype)
      complex_dtype = np.dtype(complex_dtype)

      key_2 = libjax.random.PRNGKey(seed + 1)

      real_part = libjax.random.normal(key, shape, dtype=real_dtype)
      complex_part = libjax.random.normal(key_2, shape, dtype=real_dtype)
      unit = (
          np.complex64(1j)
          if complex_dtype == np.dtype(np.complex64) else np.complex128(1j))
      return real_part + unit * complex_part

    if np.dtype(dtype) is np.dtype(jnp.complex128):
      return cmplx_randn(dtype, jnp.float64)
    if np.dtype(dtype) is np.dtype(jnp.complex64):
      return cmplx_randn(dtype, jnp.float32)

    return libjax.random.normal(key, shape).astype(dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[np.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:
    if not seed:
      seed = np.random.randint(0, 2**63)
    key = libjax.random.PRNGKey(seed)

    dtype = dtype if dtype is not None else np.dtype(np.float64)

    def cmplx_random_uniform(complex_dtype, real_dtype):
      real_dtype = np.dtype(real_dtype)
      complex_dtype = np.dtype(complex_dtype)

      key_2 = libjax.random.PRNGKey(seed + 1)

      real_part = libjax.random.uniform(
          key,
          shape,
          dtype=real_dtype,
          minval=boundaries[0],
          maxval=boundaries[1])
      complex_part = libjax.random.uniform(
          key_2,
          shape,
          dtype=real_dtype,
          minval=boundaries[0],
          maxval=boundaries[1])
      unit = (
          np.complex64(1j)
          if complex_dtype == np.dtype(np.complex64) else np.complex128(1j))
      return real_part + unit * complex_part

    if np.dtype(dtype) is np.dtype(jnp.complex128):
      return cmplx_random_uniform(dtype, jnp.float64)
    if np.dtype(dtype) is np.dtype(jnp.complex64):
      return cmplx_random_uniform(dtype, jnp.float32)

    return libjax.random.uniform(
        key, shape, minval=boundaries[0], maxval=boundaries[1]).astype(dtype)

  def eigsh_lanczos(
      self,
      A: Callable,
      args: List[Tensor],
      initial_state: Optional[Tensor] = None,
      shape: Optional[Tuple] = None,
      dtype: Optional[Type[np.number]] = None,
      num_krylov_vecs: int = 20,
      numeig: int = 1,
      tol: float = 1E-8,
      delta: float = 1E-8,
      ndiag: int = 10,
      reorthogonalize: Optional[bool] = False) -> Tuple[List, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`. `A` is a function implementing the matrix-vector
    product. 
    WARNING: This routine uses jax.jit to reduce runtimes. jitting is triggered
    at the first invocation of `eigsh_lanczos`, and on any subsequent calls 
    if the python `id` of `A` changes, even if the formal definition of `A` 
    stays the same. 
    Example: the following will jit once at the beginning, and then never again:

    ```python
    import jax
    import numpy as np
    def A(H,x):
      return jax.np.dot(H,x)
    for n in range(100):
      H = jax.np.array(np.random.rand(10,10))
      x = jax.np.array(np.random.rand(10,10))
      res = eigsh_lanczos(A, [H],x) #jitting is triggerd only at `n=0`
    ```

    The following code triggers jitting at every iteration, which 
    results in considerably reduced performance

    ```python
    import jax
    import numpy as np
    for n in range(100):
      def A(H,x):
        return jax.np.dot(H,x)
      H = jax.np.array(np.random.rand(10,10))
      x = jax.np.array(np.random.rand(10,10))
      res = eigsh_lanczos(A, [H],x) #jitting is triggerd at every step `n`
    ```
    
    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(*args, vector)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(*args, initial_state)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If both no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
        `np.linalg.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
        as stopping criterion between two diagonalization steps of the
        tridiagonal operator.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. It means that an (approximate) invariant subspace has
        been found.
      ndiag: The tridiagonal Operator is diagonalized every `ndiag` iterations
        to check convergence.
      reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
        explicit orthogonalization (more costly than `reorthogonalize=False`)
    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
    if num_krylov_vecs < numeig:
      raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if numeig > 1 and not reorthogonalize:
      raise ValueError(
          "Got numeig = {} > 1 and `reorthogonalize = False`. "
          "Use `reorthogonalize=True` for `numeig > 1`".format(numeig))
    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, jnp.ndarray):
      raise TypeError("Expected a `jax.array`. Got {}".format(
          type(initial_state)))
    if A not in _CACHED_MATVECS:
      _CACHED_MATVECS[A] = libjax.tree_util.Partial(A)
    if not hasattr(self, '_jaxlan'):
      # pylint: disable=attribute-defined-outside-init
      self._jaxlan = jitted_functions._generate_jitted_eigsh_lanczos(libjax)

    return self._jaxlan(_CACHED_MATVECS[A], args, initial_state,
                        num_krylov_vecs, numeig, delta, reorthogonalize)

  def conj(self, tensor: Tensor) -> Tensor:
    return jnp.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return jnp.linalg.eigh(matrix)

  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 - tensor2

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 / tensor2

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    return libjax.ops.index_update(tensor, mask, assignee)

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to numpy backend method `inv` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    return jnp.linalg.inv(matrix)

  def broadcast_right_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor2.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`,"
                       " found `tensor2.shape = {}`".format(tensor2.shape))
    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor, tensor2: Tensor):
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

  def sin(self, tensor: Tensor):
    return jnp.sin(tensor)

  def cos(self, tensor: Tensor):
    return jnp.cos(tensor)

  def exp(self, tensor: Tensor):
    return jnp.exp(tensor)

  def log(self, tensor: Tensor):
    return jnp.log(tensor)

  def expm(self, matrix: Tensor):
    if len(matrix.shape) != 2:
      raise ValueError("input to numpy backend method `expm` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("input to numpy backend method `expm` only supports"
                       " N*N matrix, {x}*{y} matrix is given".format(
                           x=matrix.shape[0], y=matrix.shape[1]))
    # pylint: disable=no-member
    return jsp.linalg.expm(matrix)
