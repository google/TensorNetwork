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

  def _generate_jitted_eigsh_lanczos(self):
    """
    Helper function to generate jitted lanczos function used in eigsh_lanczos.
    """

    @partial(self.jax.jit, static_argnums=(3, 4, 5, 6))
    def jax_lanczos(matvec, arguments, init, ncv, neig, landelta, reortho):

      def body_reortho(i, vals):
        vector, krylov_vectors = vals
        v = krylov_vectors[i, :]
        vector -= self.jax.numpy.dot(
            self.jax.numpy.conj(v),
            self.jax.numpy.ravel(vector)) * self.jax.numpy.reshape(
                v, vector.shape)
        return [vector, krylov_vectors]

      def body_lanczos(vals):
        current_vector, krylov_vectors, vector_norms, diagonal_elements, matvec, args, _, threshold, i, maxiteration = vals
        #current_vector = krylov_vectors[i,:]
        norm = self.jax.numpy.linalg.norm(self.jax.numpy.ravel(current_vector))
        normalized_vector = current_vector / norm
        normalized_vector, krylov_vectors = self.jax.lax.cond(
            reortho, True, lambda x: self.jax.lax.fori_loop(
                0, i, body_reortho, [normalized_vector, krylov_vectors]),
            False, lambda x: [normalized_vector, krylov_vectors])
        Av = matvec(*args, normalized_vector)

        diag_element = self.jax.numpy.dot(
            self.jax.numpy.conj(self.jax.numpy.ravel(normalized_vector)),
            self.jax.numpy.ravel(Av))

        res = self.jax.numpy.reshape(
            self.jax.numpy.ravel(Av) -
            self.jax.numpy.ravel(normalized_vector) * diag_element -
            krylov_vectors[i - 1] * norm, Av.shape)
        krylov_vectors = self.jax.ops.index_update(
            krylov_vectors, self.jax.ops.index[i, :],
            self.jax.numpy.ravel(normalized_vector))

        vector_norms = self.jax.ops.index_update(
            vector_norms, self.jax.ops.index[i - 1], norm)
        diagonal_elements = self.jax.ops.index_update(
            diagonal_elements, self.jax.ops.index[i - 1], diag_element)

        return [
            res, krylov_vectors, vector_norms, diagonal_elements, matvec, args,
            norm, threshold, i + 1, maxiteration
        ]

      def cond_fun(vals):
        _, _, _, _, _, _, norm, threshold, iteration, maxiteration = vals

        def check_thresh(check_vals):
          val, thresh = check_vals
          return self.jax.lax.cond(val < thresh, False, lambda x: x,
                                   True, lambda x: x)

        return self.jax.lax.cond(iteration <= maxiteration, [norm, threshold],
                                 check_thresh, False, lambda x: x)

      numel = self.jax.numpy.prod(init.shape)
      krylov_vecs = self.jax.numpy.zeros((ncv + 1, numel))

      norms = self.jax.numpy.zeros(ncv)
      diag_elems = self.jax.numpy.zeros(ncv)

      norm = self.jax.numpy.linalg.norm(init)
      norms = self.jax.ops.index_update(norms, self.jax.ops.index[0], 1.0)

      initvals = [
          init, krylov_vecs, norms, diag_elems, matvec, arguments, 1.0,
          landelta, 1, ncv
      ]

      final_state, krylov_vecs, norms, diags, _, _, _, _, it, _ = self.jax.lax.while_loop(
          cond_fun, body_lanczos, initvals)
      krylov_vecs = self.jax.ops.index_update(krylov_vecs,
                                              self.jax.ops.index[it, :],
                                              self.jax.numpy.ravel(final_state))
      A_tridiag = self.jax.numpy.diag(diags) + self.jax.numpy.diag(
          norms[1:], 1) + self.jax.numpy.diag(
              self.jax.numpy.conj(norms[1:]), -1)
      eigvals, U = self.jax.numpy.linalg.eigh(A_tridiag)
      eigvals = eigvals.astype(A_tridiag.dtype)

      def body_vector(i, vals):
        krv, unitary, states = vals
        dim = unitary.shape[1]
        n, m = self.jax.numpy.divmod(i, dim)
        states = self.jax.ops.index_add(states, self.jax.ops.index[n],
                                        krv[m + 1, :] * unitary[m, n])
        return [krv, unitary, states]

      state_vector = self.jax.numpy.zeros([neig, numel])
      _, _, vector = self.jax.lax.fori_loop(
          0, neig * (krylov_vecs.shape[0] - 1), body_vector,
          [krylov_vecs, U, state_vector])
      vector /= self.jax.numpy.linalg.norm(vector)
      return self.jax.numpy.array(eigvals[0:neig]), [
          vector[n, :] / self.jax.numpy.linalg.norm(
              self.jax.numpy.ravel(vector[n, :])) for n in range(neig)
      ], krylov_vecs

    return jax_lanczos

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
      self._jaxlan = self._generate_jitted_eigsh_lanczos()

    return self._jaxlan(A, args, initial_state, num_krylov_vecs, numeig, delta,
                        reorthogonalize)

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    return self.jax.ops.index_update(tensor, mask, assignee)

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    return self.jax.jit(fun, **kwargs)
