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
from typing import Union
from tensornetwork.backends import abstract_backend
from tensornetwork.backends.numpy import decompositions
import numpy as np
from tensornetwork.backends.jax import jitted_functions
from functools import partial
import warnings

Tensor = Any
# pylint: disable=abstract-method

_CACHED_MATVECS = {}
_CACHED_FUNCTIONS = {}

class JaxBackend(abstract_backend.AbstractBackend):
  """See abstract_backend.AbstractBackend for documentation."""

  def __init__(self, dtype: Optional[np.dtype] = None,
               precision: Optional[Text] = None) -> None:
    # pylint: disable=global-variable-undefined
    global libjax  # Jax module
    global jnp  # jax.numpy module
    global jsp  # jax.scipy module
    super().__init__()
    try:
      #pylint: disable=import-outside-toplevel
      import jax
    except ImportError as err:
      raise ImportError("Jax not installed, please switch to a different "
                        "backend or install Jax.") from err
    libjax = jax
    jnp = libjax.numpy
    jsp = libjax.scipy
    self.name = "jax"
    self._dtype = np.dtype(dtype) if dtype is not None else None
    self.jax_precision = precision if precision is not None else libjax.lax.Precision.DEFAULT #pylint: disable=line-too-long

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    return jnp.tensordot(a, b, axes, precision=self.jax_precision)

  def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
    return jnp.reshape(tensor, np.asarray(shape).astype(np.int32))

  def transpose(self, tensor, perm=None) -> Tensor:
    return jnp.transpose(tensor, perm)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    return libjax.lax.dynamic_slice(tensor, start_indices, slice_sizes)

  def svd(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd(
        jnp,
        tensor,
        pivot_axis,
        max_singular_values,
        max_truncation_error,
        relative=relative)

  def qr(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      non_negative_diagonal: bool = False
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.qr(jnp, tensor, pivot_axis, non_negative_diagonal)

  def rq(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      non_negative_diagonal: bool = False
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq(jnp, tensor, pivot_axis, non_negative_diagonal)


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

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if (not isinstance(tensor, jnp.ndarray) and not jnp.isscalar(tensor)):
      raise TypeError("Expected a `jnp.array` or scalar. Got {}".format(
          type(tensor)))
    result = jnp.asarray(tensor)
    return result

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return jnp.tensordot(tensor1, tensor2, 0,
                         precision=self.jax_precision)

  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
    return jnp.einsum(expression, *tensors, optimize=optimize)

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

  def eigs(self, #pylint: disable=arguments-differ
           A: Callable,
           args: Optional[List] = None,
           initial_state: Optional[Tensor] = None,
           shape: Optional[Tuple[int, ...]] = None,
           dtype: Optional[Type[np.number]] = None,
           num_krylov_vecs: int = 50,
           numeig: int = 6,
           tol: float = 1E-8,
           which: Text = 'LR',
           maxiter: int = 20) -> Tuple[Tensor, List]:
    """
    Implicitly restarted Arnoldi method for finding the lowest
    eigenvector-eigenvalue pairs of a linear operator `A`.
    `A` is a function implementing the matrix-vector
    product.

    WARNING: This routine uses jax.jit to reduce runtimes. jitting is triggered
    at the first invocation of `eigs`, and on any subsequent calls
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
      res = eigs(A, [H],x) #jitting is triggerd only at `n=0`
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
      res = eigs(A, [H],x) #jitting is triggerd at every step `n`
    ```

    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(vector, *args)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The number of eigenvector-eigenvalue pairs to be computed.
      tol: The desired precision of the eigenvalues. For the jax backend
        this has currently no effect, and precision of eigenvalues is not
        guaranteed. This feature may be added at a later point. To increase
        precision the caller can either increase `maxiter` or `num_krylov_vecs`.
      which: Flag for targetting different types of eigenvalues. Currently
        supported are `which = 'LR'` (larges real part) and `which = 'LM'`
        (larges magnitude).
      maxiter: Maximum number of restarts. For `maxiter=0` the routine becomes
        equivalent to a simple Arnoldi method.
    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` eigenvalues
       eigvecs: A list of `numeig` eigenvectors
    """

    if args is None:
      args = []
    if which not in ('LR', 'LM'):
      raise ValueError(f'which = {which} is currently not supported.')

    if numeig > num_krylov_vecs:
      raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, jnp.ndarray):
      raise TypeError("Expected a `jax.array`. Got {}".format(
          type(initial_state)))

    if A not in _CACHED_MATVECS:
      _CACHED_MATVECS[A] = libjax.tree_util.Partial(libjax.jit(A))

    if "imp_arnoldi" not in _CACHED_FUNCTIONS:
      imp_arnoldi = jitted_functions._implicitly_restarted_arnoldi(libjax)
      _CACHED_FUNCTIONS["imp_arnoldi"] = imp_arnoldi

    eta, U, numits = _CACHED_FUNCTIONS["imp_arnoldi"](_CACHED_MATVECS[A], args,
                                                      initial_state,
                                                      num_krylov_vecs, numeig,
                                                      which, tol, maxiter,
                                                      self.jax_precision)
    if numeig > numits:
      warnings.warn(
          f"Arnoldi terminated early after numits = {numits}"
          f" < numeig = {numeig} steps. For this value of `numeig `"
          f"the routine will return spurious eigenvalues of value 0.0."
          f"Use a smaller value of numeig, or a smaller value for `tol`")
    return eta, U

  def eigsh(
      self,  #pylint: disable=arguments-differ
      A: Callable,
      args: Optional[List] = None,
      initial_state: Optional[Tensor] = None,
      shape: Optional[Tuple[int, ...]] = None,
      dtype: Optional[Type[np.number]] = None,
      num_krylov_vecs: int = 50,
      numeig: int = 6,
      tol: float = 1E-8,
      which: Text = 'SA',
      maxiter: int = 20) -> Tuple[Tensor, List]:
    """
    Implicitly restarted Lanczos method for finding the lowest
    eigenvector-eigenvalue pairs of a symmetric (hermitian) linear operator `A`.
    `A` is a function implementing the matrix-vector
    product.

    WARNING: This routine uses jax.jit to reduce runtimes. jitting is triggered
    at the first invocation of `eigsh`, and on any subsequent calls
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
      res = eigsh(A, [H],x) #jitting is triggerd only at `n=0`
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
      res = eigsh(A, [H],x) #jitting is triggerd at every step `n`
    ```

    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(vector, *args)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The number of eigenvector-eigenvalue pairs to be computed.
      tol: The desired precision of the eigenvalues. For the jax backend
        this has currently no effect, and precision of eigenvalues is not
        guaranteed. This feature may be added at a later point. To increase
        precision the caller can either increase `maxiter` or `num_krylov_vecs`.
      which: Flag for targetting different types of eigenvalues. Currently
        supported are `which = 'LR'` (larges real part) and `which = 'LM'`
        (larges magnitude).
      maxiter: Maximum number of restarts. For `maxiter=0` the routine becomes
        equivalent to a simple Arnoldi method.
    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` eigenvalues
       eigvecs: A list of `numeig` eigenvectors
    """

    if args is None:
      args = []
    if which not in ('SA', 'LA', 'LM'):
      raise ValueError(f'which = {which} is currently not supported.')

    if numeig > num_krylov_vecs:
      raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, jnp.ndarray):
      raise TypeError("Expected a `jax.array`. Got {}".format(
          type(initial_state)))

    if A not in _CACHED_MATVECS:
      _CACHED_MATVECS[A] = libjax.tree_util.Partial(libjax.jit(A))

    if "imp_lanczos" not in _CACHED_FUNCTIONS:
      imp_lanczos = jitted_functions._implicitly_restarted_lanczos(libjax)
      _CACHED_FUNCTIONS["imp_lanczos"] = imp_lanczos

    eta, U, numits = _CACHED_FUNCTIONS["imp_lanczos"](_CACHED_MATVECS[A], args,
                                                      initial_state,
                                                      num_krylov_vecs, numeig,
                                                      which, tol, maxiter,
                                                      self.jax_precision)
    if numeig > numits:
      warnings.warn(
          f"Arnoldi terminated early after numits = {numits}"
          f" < numeig = {numeig} steps. For this value of `numeig `"
          f"the routine will return spurious eigenvalues of value 0.0."
          f"Use a smaller value of numeig, or a smaller value for `tol`")
    return eta, U

  def eigsh_lanczos(
      self,
      A: Callable,
      args: Optional[List[Tensor]] = None,
      initial_state: Optional[Tensor] = None,
      shape: Optional[Tuple] = None,
      dtype: Optional[Type[np.number]] = None,
      num_krylov_vecs: int = 20,
      numeig: int = 1,
      tol: float = 1E-8,
      delta: float = 1E-8,
      ndiag: int = 10,
      reorthogonalize: Optional[bool] = False) -> Tuple[Tensor, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a hermitian linear operator `A`. `A` is a function implementing
    the matrix-vector product.
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
         Call signature of `A` is `res = A(vector, *args)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The number of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalues. For the jax backend
        this has currently no effect, and precision of eigenvalues is not
        guaranteed. This feature may be added at a later point.
        To increase precision the caller can increase `num_krylov_vecs`.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. It means that an (approximate) invariant subspace has
        been found.
      ndiag: The tridiagonal Operator is diagonalized every `ndiag` iterations
        to check convergence. This has currently no effect for the jax backend,
        but may be added at a later point.
      reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
        explicit orthogonalization (more costly than `reorthogonalize=False`)
    Returns:
      (eigvals, eigvecs)
       eigvals: A jax-array containing `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
    if args is None:
      args = []
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
    if "eigsh_lanczos" not in _CACHED_FUNCTIONS:
      eigsh_lanczos = jitted_functions._generate_jitted_eigsh_lanczos(libjax)
      _CACHED_FUNCTIONS["eigsh_lanczos"] = eigsh_lanczos
    eigsh_lanczos = _CACHED_FUNCTIONS["eigsh_lanczos"]
    eta, U, numits = eigsh_lanczos(_CACHED_MATVECS[A], args, initial_state,
                                   num_krylov_vecs, numeig, delta,
                                   reorthogonalize, self.jax_precision)
    if numeig > numits:
      warnings.warn(
          f"Lanczos terminated early after numits = {numits}"
          f" < numeig = {numeig} steps. For this value of `numeig `"
          f"the routine will return spurious eigenvalues of value 0.0."
          f"Use a smaller value of numeig, or a smaller value for `tol`")
    return eta, U

  def _gmres(self,
             A_mv: Callable,
             b: Tensor,
             A_args: List,
             A_kwargs: dict,
             x0: Tensor,
             tol: float,
             atol: float,
             num_krylov_vectors: int,
             maxiter: int,
             M: Optional[Callable] = None) -> Tuple[Tensor, int]:
    """ GMRES solves the linear system A @ x = b for x given a vector `b` and
    a general (not necessarily symmetric/Hermitian) linear operator `A`.

    As a Krylov method, GMRES does not require a concrete matrix representation
    of the n by n `A`, but only a function
    `vector1 = A_mv(vector0, *A_args, **A_kwargs)`
    prescribing a one-to-one linear map from vector0 to vector1 (that is,
    A must be square, and thus vector0 and vector1 the same size). If `A` is a
    dense matrix, or if it is a symmetric/Hermitian operator, a different
    linear solver will usually be preferable.

    GMRES works by first constructing the Krylov basis
    K = (x0, A_mv@x0, A_mv@A_mv@x0, ..., (A_mv^num_krylov_vectors)@x_0) and then
    solving a certain dense linear system K @ q0 = q1 from whose solution x can
    be approximated. For `num_krylov_vectors = n` the solution is provably exact
    in infinite precision, but the expense is cubic in `num_krylov_vectors` so
    one is typically interested in the `num_krylov_vectors << n` case.
    The solution can in this case be repeatedly
    improved, to a point, by restarting the Arnoldi iterations each time
    `num_krylov_vectors` is reached. Unfortunately the optimal parameter choices
    balancing expense and accuracy are difficult to predict in advance, so
    applying this function requires a degree of experimentation.

    In a tensor network code one is typically interested in A_mv implementing
    some tensor contraction. This implementation thus allows `b` and `x0` to be
    of whatever arbitrary, though identical, shape `b = A_mv(x0, ...)` expects.
    Reshaping to and from a matrix problem is handled internally.

    The Jax backend version of GMRES uses a homemade implementation that, for
    now, is suboptimal for num_krylov_vecs ~ b.size.

    For the same reason as described in eigsh_lancsoz, the function A_mv
    should be Jittable (or already Jitted) and, if at all possible, defined
    only once at the global scope. A new compilation will be triggered each
    time an A_mv with a new function signature is passed in, even if the
    'new' function is identical to the old one (function identity is
    undecidable).


    Args:
      A_mv     : A function `v0 = A_mv(v, *A_args, **A_kwargs)` where `v0` and
                 `v` have the same shape.
      b        : The `b` in `A @ x = b`; it should be of the shape `A_mv`
                 operates on.
      A_args   : Positional arguments to `A_mv`, supplied to this interface
                 as a list.
                 Default: None.
      A_kwargs : In the other backends, keyword arguments to `A_mv`, supplied
                 as a dictionary. However, the Jax backend does not support
                 A_mv accepting
                 keyword arguments since this causes problems with Jit.
                 Therefore, an error is thrown if A_kwargs is specified.
                 Default: None.
      x0       : An optional guess solution. Zeros are used by default.
                 If `x0` is supplied, its shape and dtype must match those of
                 `b`, or an
                 error will be thrown.
                 Default: zeros.
      tol, atol: Solution tolerance to achieve,
                 norm(residual) <= max(tol*norm(b), atol).
                 Default: tol=1E-05
                          atol=tol
      num_krylov_vectors
               : Size of the Krylov space to build at each restart.
                 Expense is cubic in this parameter.
                 Default: 20.
      maxiter  : The Krylov space will be repeatedly rebuilt up to this many
                 times. Large values of this argument
                 should be used only with caution, since especially for nearly
                 symmetric matrices and small `num_krylov_vectors` convergence
                 might well freeze at a value significantly larger than `tol`.
                 Default: 1
      M        : Inverse of the preconditioner of A; see the docstring for
                 `scipy.sparse.linalg.gmres`. This is unsupported in the Jax
                 backend, and NotImplementedError will be raised if it is
                 supplied.
                 Default: None.


    Raises:
      ValueError: -if `x0` is supplied but its shape differs from that of `b`.
                  -if num_krylov_vectors <= 0.
                  -if tol or atol was negative.
      NotImplementedError: - If M is supplied.
                           - If A_kwargs is supplied.
      TypeError:  -if the dtype of `x0` and `b` are mismatching.
    Returns:
      x       : The converged solution. It has the same shape as `b`.
      info    : 0 if convergence was achieved, the number of restarts otherwise.
    """

    if M is not None:
      raise NotImplementedError("M is not supported by the Jax backend.")
    if A_kwargs:
      raise NotImplementedError("A_kwargs is not supported by the Jax backend.")


    if A_mv not in _CACHED_MATVECS:
      @libjax.tree_util.Partial
      def matrix_matvec(x, *args):
        x = x.reshape(b.shape)
        result = A_mv(x, *args)
        return result.ravel()
      _CACHED_MATVECS[A_mv] = matrix_matvec

    if "gmres" not in _CACHED_FUNCTIONS:
      _CACHED_FUNCTIONS["gmres"] = jitted_functions.gmres_wrapper(libjax)
    gmres_m = _CACHED_FUNCTIONS["gmres"].gmres_m
    x, _, n_iter, converged = gmres_m(_CACHED_MATVECS[A_mv], A_args, b.ravel(),
                                      x0, tol, atol, num_krylov_vectors,
                                      maxiter, self.jax_precision)
    if converged:
      info = 0
    else:
      info = n_iter
    x = self.reshape(x, b.shape)
    return x, info

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

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    if len(tensor2.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`,"
                       " found `tensor2.shape = {}`".format(tensor2.shape))
    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

  def sin(self, tensor: Tensor) -> Tensor:
    return jnp.sin(tensor)

  def cos(self, tensor: Tensor) -> Tensor:
    return jnp.cos(tensor)

  def exp(self, tensor: Tensor) -> Tensor:
    return jnp.exp(tensor)

  def log(self, tensor: Tensor) -> Tensor:
    return jnp.log(tensor)

  def expm(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) != 2:
      raise ValueError("input to numpy backend method `expm` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("input to numpy backend method `expm` only supports"
                       " N*N matrix, {x}*{y} matrix is given".format(
                           x=matrix.shape[0], y=matrix.shape[1]))
    # pylint: disable=no-member
    return jsp.linalg.expm(matrix)

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    return libjax.jit(fun, *args, **kwargs)

  def sum(self,
          tensor: Tensor,
          axis: Optional[Sequence[int]] = None,
          keepdims: bool = False) -> Tensor:
    return jnp.sum(tensor, axis=axis, keepdims=keepdims)

  def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if (tensor1.ndim <= 1) or (tensor2.ndim <= 1):
      raise ValueError("inputs to `matmul` have to be tensors of order > 1,")
    return jnp.matmul(tensor1, tensor2, precision=self.jax_precision)

  def diagonal(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
               axis2: int = -1) -> Tensor:
    """Return specified diagonals.

    If tensor is 2-D, returns the diagonal of tensor with the given offset,
    i.e., the collection of elements of the form a[i, i+offset].
    If a has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
    returned. The shape of the resulting array can be determined by removing
    axis1 and axis2 and appending an index to the right equal to the size of the
    resulting diagonals.

    This function only extracts diagonals. If you
    wish to create diagonal matrices from vectors, use diagflat.

    Args:
      tensor: A tensor.
      offset: Offset of the diagonal from the main diagonal.
      axis1, axis2: Axis to be used as the first/second axis of the 2D
                    sub-arrays from which the diagonals should be taken.
                    Defaults to second last/last axis.
    Returns:
      array_of_diagonals: A dim = min(1, tensor.ndim - 2) tensor storing
                          the batched diagonals.
    """
    if axis1 == axis2:
      raise ValueError("axis1, axis2 cannot be equal.")
    return jnp.diagonal(tensor, offset=offset, axis1=axis1, axis2=axis2)

  def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
    """ Flattens tensor and creates a new matrix of zeros with its elements
    on the k'th diagonal.
    Args:
      tensor: A tensor.
      k     : The diagonal upon which to place its elements.
    Returns:
      tensor: A new tensor with all zeros save the specified diagonal.
    """
    return jnp.diag(jnp.ravel(tensor), k=k)

  def trace(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
    """Return summed entries along diagonals.

    If tensor is 2-D, the sum is over the
    diagonal of tensor with the given offset,
    i.e., the collection of elements of the form a[i, i+offset].
    If a has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
    summed.

    Args:
      tensor: A tensor.
      offset: Offset of the diagonal from the main diagonal.
      axis1, axis2: Axis to be used as the first/second axis of the 2D
                    sub-arrays from which the diagonals should be taken.
                    Defaults to second last/last axis.
    Returns:
      array_of_diagonals: The batched summed diagonals.
    """
    if axis1 == axis2:
      raise ValueError("axis1, axis2 cannot be equal.")
    return jnp.trace(tensor, offset=offset, axis1=axis1, axis2=axis2)

  def abs(self, tensor: Tensor) -> Tensor:
    """
    Returns the elementwise absolute value of tensor.
    Args:
      tensor: An input tensor.
    Returns:
      tensor: Its elementwise absolute value.
    """
    return jnp.abs(tensor)

  def sign(self, tensor: Tensor) -> Tensor:
    """
    Returns an elementwise tensor with entries
    y[i] = 1, 0, -1 where tensor[i] > 0, == 0, and < 0 respectively.

    For complex input the behaviour of this function may depend on the backend.
    The Jax backend version returns y[i] = x[i]/sqrt(x[i]^2).

    Args:
      tensor: The input tensor.
    """
    return jnp.sign(tensor)

  def item(self, tensor):
    return tensor.item()

  def power(self, a: Tensor, b: Union[Tensor, float]) -> Tensor:
    """
    Returns the power of tensor a to the value of b.
    In the case b is a tensor, then the power is by element
      with a as the base and b as the exponent.
    In the case b is a scalar, then the power of each value in a
      is raised to the exponent of b.

    Args:
      a: The tensor that contains the base.
      b: The tensor that contains the exponent or a single scalar.
    """
    return jnp.power(a, b)
