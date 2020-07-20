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
#pyling: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Callable, List, Text, Type
from tensornetwork.backends import abstract_backend
from tensornetwork.backends.symmetric import decompositions
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
import tensornetwork.block_sparse as bs
import numpy
Tensor = Any

# TODO (mganahl): implement eigs


# pylint: disable=abstract-method
class SymmetricBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    super(SymmetricBackend, self).__init__()
    self.bs = bs
    self.name = "symmetric"

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Sequence[Sequence[int]]) -> Tensor:
    return self.bs.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
    return self.bs.reshape(tensor, numpy.asarray(shape).astype(numpy.int32))

  def transpose(self, tensor, perm) -> Tensor:
    return self.bs.transpose(tensor, perm)

  def svd(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd(self.bs, tensor, pivot_axis, max_singular_values,
                              max_truncation_error, relative)

  def qr(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      non_negative_diagonal: bool = False
  ) -> Tuple[Tensor, Tensor]:
    if non_negative_diagonal:
      errstr = "Can't specify non_negative_diagonal with BlockSparse."
      raise NotImplementedError(errstr)
    return decompositions.qr(self.bs, tensor, pivot_axis)

  def rq(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      non_negative_diagonal: bool = False
  ) -> Tuple[Tensor, Tensor]:
    if non_negative_diagonal:
      errstr = "Can't specify non_negative_diagonal with BlockSparse."
      raise NotImplementedError(errstr)
    return decompositions.rq(self.bs, tensor, pivot_axis)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return numpy.concatenate(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.sparse_shape

  def shape_prod(self, values: Tensor) -> Tensor:
    return numpy.prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return self.bs.sqrt(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if numpy.isscalar(tensor):
      tensor = BlockSparseTensor(
          data=tensor, charges=[], flows=[], order=[], check_consistency=False)

    if not isinstance(tensor, BlockSparseTensor):
      raise TypeError(
          "cannot convert tensor of type `{}` to `BlockSparseTensor`".format(
              type(tensor)))
    return tensor

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.bs.tensordot(tensor1, tensor2, 0)

  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
    raise NotImplementedError("`einsum` currently not implemented")

  def norm(self, tensor: Tensor) -> Tensor:
    return self.bs.norm(tensor)

  def eye(self,
          N: Index,
          dtype: Optional[numpy.dtype] = None,
          M: Optional[Index] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64

    return self.bs.eye(N, M, dtype=dtype)

  def ones(self,
           shape: Sequence[Index],
           dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64
    return self.bs.ones(shape, dtype=dtype)

  def zeros(self,
            shape: Sequence[Index],
            dtype: Optional[numpy.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else numpy.float64
    return self.bs.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Sequence[Index],
            dtype: Optional[numpy.dtype] = None,
            seed: Optional[int] = None) -> Tensor:

    if seed:
      numpy.random.seed(seed)
    return self.bs.randn(shape, dtype)

  def random_uniform(self,
                     shape: Sequence[Index],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[numpy.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:

    if seed:
      numpy.random.seed(seed)
    dtype = dtype if dtype is not None else numpy.float64
    return self.bs.random(shape, boundaries, dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return self.bs.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return self.bs.eigh(matrix)

  def eigsh_lanczos(self, #pylint: disable=arguments-differ
                    A: Callable,
                    args: Optional[List[Tensor]] = None,
                    initial_state: Optional[Tensor] = None,
                    shape: Optional[Tuple] = None,
                    dtype: Optional[Type[numpy.number]] = None,
                    num_krylov_vecs: int = 20,
                    numeig: int = 1,
                    tol: float = 1E-8,
                    delta: float = 1E-8,
                    ndiag: int = 20,
                    reorthogonalize: bool = False,
                    enable_caching: bool = True) -> Tuple[Tensor, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`.
    Note: read notes for `enable_caching` carefully.
    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(vector, *args)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If both no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
        `numpy.linalg.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
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
      enable_caching: If `True`, block-data during calls to `matvec` is cached
        for later reuse. Note: usually it is save to enable_caching, unless 
        `matvec` uses matrix decompositions liek SVD, QR, eigh, eig or similar.
        In this case, if one does a large number of krylov steps, this can lead 
        to memory clutter and/or overflow.

    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
    former_caching_status = self.bs.get_caching_status()
    self.bs.set_caching_status(enable_caching)
    if enable_caching:
      cache_was_empty = self.bs.get_cacher().is_empty

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

    if not isinstance(initial_state, BlockSparseTensor):
      raise TypeError("Expected a `BlockSparseTensor`. Got {}".format(
          type(initial_state)))

    vector_n = initial_state
    vector_n.contiguous() # bring into contiguous memory layout

    Z = self.norm(vector_n)
    vector_n /= Z
    norms_vector_n = []
    diag_elements = []
    krylov_vecs = []
    first = True
    eigvalsold = []

    for it in range(num_krylov_vecs):
      # normalize the current vector:
      norm_vector_n = self.norm(vector_n)
      if abs(norm_vector_n) < delta:
        # we found an invariant subspace, time to stop
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      # store the Lanczos vector for later
      if reorthogonalize:
        # vector_n is always in contiguous memory layout at this point
        for v in krylov_vecs:
          v.contiguous() # make sure storage layouts are matching
          # it's save to operate on the tensor data now (pybass some checks)
          vector_n.data -= numpy.dot(numpy.conj(v.data), vector_n.data) * v.data
      krylov_vecs.append(vector_n)
      A_vector_n = A(vector_n, *args)
      A_vector_n.contiguous() # contiguous memory layout

      # operate on tensor-data for scalar products
      # this can be potentially problematic if vector_n and A_vector_n
      # have non-matching shapes due to an erroneous matvec.
      # If this is the case though an error will be thrown at line 281
      diag_elements.append(
          numpy.dot(numpy.conj(vector_n.data), A_vector_n.data))

      if (it > 0) and (it % ndiag == 0) and (len(diag_elements) >= numeig):
        # diagonalize the effective Hamiltonian
        A_tridiag = numpy.diag(diag_elements) + numpy.diag(
            norms_vector_n[1:], 1) + numpy.diag(
                numpy.conj(norms_vector_n[1:]), -1)
        eigvals, u = numpy.linalg.eigh(A_tridiag)
        if not first:
          if numpy.linalg.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = numpy.diag(diag_elements) + numpy.diag(
        norms_vector_n[1:], 1) + numpy.diag(numpy.conj(norms_vector_n[1:]), -1)
    eigvals, u = numpy.linalg.eigh(A_tridiag)
    eigenvectors = []
    eigvals = numpy.array(eigvals).astype(A_tridiag.dtype)

    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.sparse_shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / self.norm(state))
      
    self.bs.set_caching_status(former_caching_status)      
    if enable_caching and cache_was_empty:
      self.bs.clear_cache()
        
    return eigvals[0:numeig], eigenvectors

  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 - tensor2

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 * tensor2

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 / tensor2

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to symmetric backend method `inv` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    return self.bs.inv(matrix)

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    if tensor2.ndim != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor2`,"
                       " found `tensor2.shape = {}`".format(tensor2.shape))
    return self.tensordot(tensor1, self.diagflat(tensor2),
                          ([len(tensor1.shape) - 1], [0]))

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))
    return self.tensordot(self.diagflat(tensor1), tensor2, ([1], [0]))

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    return fun

  def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
    if k != 0:
      raise NotImplementedError("Can't specify k with Symmetric backend")
    return self.bs.diag(tensor)

  def diagonal(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
               axis2: int = -1) -> Tensor:
    if axis1 != -2 or axis2 != -1 or offset != 0:
      errstr = "offset, axis1, axis2 unsupported by Symmetric backend."
      raise NotImplementedError(errstr)
    return self.bs.diag(tensor)

  def trace(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
    # Default np.trace uses first two axes.
    if offset != 0:
      errstr = f"offset = {offset} must be 0 with Symmetric backend."
      raise NotImplementedError(errstr)
    if axis1 == axis2:
      raise ValueError(f"axis1 = {axis1} cannot equal axis2 = {axis2}")
    return self.bs.trace(tensor, (axis1, axis2))

  def pivot(self, tensor: Tensor, pivot_axis: int = 1) -> Tensor:
    raise NotImplementedError("Symmetric backend doesn't support pivot.")
