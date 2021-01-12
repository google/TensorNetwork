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
from typing import Union
from tensornetwork.backends import abstract_backend
from tensornetwork.backends.symmetric import decompositions
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import (BlockSparseTensor,
                                                          ChargeArray)
import warnings
import scipy as sp
import scipy.sparse.linalg
import tensornetwork.block_sparse as bs
import numpy
Tensor = Any

# pylint: disable=abstract-method
class SymmetricBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    super().__init__()
    self.bs = bs
    self.name = "symmetric"

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    return self.bs.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
    return self.bs.reshape(tensor, numpy.asarray(shape).astype(numpy.int32))

  def transpose(self, tensor, perm=None) -> Tensor:
    if perm is None:
      perm = tuple(range(tensor.ndim - 1, -1, -1))
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

    if not isinstance(tensor, ChargeArray):
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

  def norm(self, tensor: Tensor) -> float:
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

  def eigs(self,#pylint: disable=arguments-differ
           A: Callable,
           args: Optional[List] = None,
           initial_state: Optional[Tensor] = None,
           shape: Optional[Tuple[Index, ...]] = None,
           dtype: Optional[Type[numpy.number]] = None,
           num_krylov_vecs: int = 50,
           numeig: int = 6,
           tol: float = 1E-8,
           which: Text = 'LR',
           maxiter: Optional[int] = None,
           enable_caching: bool = True) -> Tuple[Tensor, List]:
    """
    Arnoldi method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`.
    If no `initial_state` is provided then `shape`  and `dtype` are required
    so that a suitable initial state can be randomly generated.
    This is a wrapper for scipy.sparse.linalg.eigs which only supports
    a subset of the arguments of scipy.sparse.linalg.eigs.
    Note: read notes for `enable_caching` carefully.

    Args:
      A: A (sparse) implementation of a linear operator
      args: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the algorithm. If `None`,
        a random initial `Tensor` is created using the `numpy.random.randn`
        method.
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If both no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
      which : ['LM' | 'SM' | 'LR' | 'SR' | 'LI']
        Which `k` eigenvectors and eigenvalues to find:
            'LM' : largest magnitude
            'SM' : smallest magnitude
            'LR' : largest real part
            'SR' : smallest real part
            'LI' : largest imaginary part
      maxiter: The maximum number of iterations.
      enable_caching: If `True`, block-data during calls to `matvec` are cached
        for later reuse. Note: usually it is save to enable_caching, unless
        `matvec` uses matrix decompositions like SVD, QR, eigh, eig or similar.
        In this case, if one does a large number of krylov steps, this can lead
        to memory clutter and/or overflow.

    Returns:
       `np.ndarray`: An array of `numeig` lowest eigenvalues
       `list`: A list of `numeig` lowest eigenvectors
    """

    if args is None:
      args = []

    if which in ('SI', 'LI'):
      raise ValueError(f'which = {which} is currently not supported.')

    if numeig + 1 >= num_krylov_vecs:
      raise ValueError("`num_krylov_vecs` > `numeig + 1` required")

    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, BlockSparseTensor):
      raise TypeError("Expected a `BlockSparseTensor`. Got {}".format(
          type(initial_state)))
    
    initial_state.contiguous(inplace=True)
    dim = len(initial_state.data)
    def matvec(vector):
      tmp.data = vector
      res = A(tmp, *args)
      res.contiguous(inplace=True)
      return res.data
    tmp = BlockSparseTensor(
        numpy.empty(0, dtype=initial_state.dtype),
        initial_state._charges,
        initial_state._flows,
        check_consistency=False)
    lop = sp.sparse.linalg.LinearOperator(
        dtype=initial_state.dtype, shape=(dim, dim), matvec=matvec)

    former_caching_status = self.bs.get_caching_status()
    self.bs.set_caching_status(enable_caching)
    if enable_caching:
      cache_was_empty = self.bs.get_cacher().is_empty
    try:
      eta, U = sp.sparse.linalg.eigs(
          A=lop,
          k=numeig,
          which=which,
          v0=initial_state.data,
          ncv=num_krylov_vecs,
          tol=tol,
          maxiter=maxiter)
    finally:
      #set caching status back to what it was
      self.bs.set_caching_status(former_caching_status)
      if enable_caching and cache_was_empty:
        self.bs.clear_cache()

    eVs = [
        BlockSparseTensor(
            U[:, n],
            initial_state._charges,
            initial_state._flows,
            check_consistency=False) for n in range(numeig)
    ]

    self.bs.set_caching_status(former_caching_status)
    if enable_caching and cache_was_empty:
      self.bs.clear_cache()

    return eta, eVs

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
        for later reuse. Note: usually it is safe to enable_caching, unless
        `matvec` uses matrix decompositions like SVD, QR, eigh, eig or similar.
        In this case, if one does a large number of krylov steps, this can lead
        to memory clutter and/or OOM errors.

    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` lowest eigenvalues
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

    if not isinstance(initial_state, BlockSparseTensor):
      raise TypeError("Expected a `BlockSparseTensor`. Got {}".format(
          type(initial_state)))

    former_caching_status = self.bs.get_caching_status()
    self.bs.set_caching_status(enable_caching)
    if enable_caching:
      cache_was_empty = self.bs.get_cacher().is_empty
    try:
      vector_n = initial_state
      vector_n.contiguous(inplace=True)  # bring into contiguous memory layout

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
            v.contiguous(inplace=True)  # make sure storage layouts are matching
            # it's save to operate on the tensor data now (pybass some checks)
            vector_n.data -= numpy.dot(numpy.conj(v.data),
                                       vector_n.data) * v.data
        krylov_vecs.append(vector_n)
        A_vector_n = A(vector_n, *args)
        A_vector_n.contiguous(inplace=True)  # contiguous memory layout

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
            if numpy.linalg.norm(eigvals[0:numeig] -
                                 eigvalsold[0:numeig]) < tol:
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
          norms_vector_n[1:], 1) + numpy.diag(
              numpy.conj(norms_vector_n[1:]), -1)
      eigvals, u = numpy.linalg.eigh(A_tridiag)
      eigenvectors = []
      eigvals = numpy.array(eigvals).astype(A_tridiag.dtype)

      for n2 in range(min(numeig, len(eigvals))):
        state = self.zeros(initial_state.sparse_shape, initial_state.dtype)
        for n1, vec in enumerate(krylov_vecs):
          state += vec * u[n1, n2]
        eigenvectors.append(state / self.norm(state))

    finally:
      # reset caching status to what it was in case of
      # and exception
      self.bs.set_caching_status(former_caching_status)
      if enable_caching and cache_was_empty:
        self.bs.clear_cache()

    return eigvals[0:numeig], eigenvectors

  def gmres(self,#pylint: disable=arguments-differ
            A_mv: Callable,
            b: BlockSparseTensor,
            A_args: Optional[List] = None,
            A_kwargs: Optional[dict] = None,
            x0: Optional[BlockSparseTensor] = None,
            tol: float = 1E-05,
            atol: Optional[float] = None,
            num_krylov_vectors: Optional[int] = None,
            maxiter: Optional[int] = 1,
            M: Optional[Callable] = None,
            enable_caching: bool = True) -> Tuple[BlockSparseTensor, int]:
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

    The numpy backend version of GMRES is simply an interface to
    `scipy.sparse.linalg.gmres`, itself an interace to ARPACK.
    SciPy 1.1.0 or newer (May 05 2018) is required.

    Args:
      A_mv: A function `v0 = A_mv(v, *A_args, **A_kwargs)` where `v0` and
        `v` have the same shape.
      b: The `b` in `A @ x = b`; it should be of the shape `A_mv`
        operates on.
      A_args: Positional arguments to `A_mv`, supplied to this interface
        as a list. Default: None.
      A_kwargs: Keyword arguments to `A_mv`, supplied to this interface
        as a dictionary.
                 Default: None.
      x0: An optional guess solution. Zeros are used by default.
        If `x0` is supplied, its shape and dtype must match those of
        b`, or an error will be thrown. Default: zeros.
      tol, atol: Solution tolerance to achieve, 
        norm(residual) <= max(tol*norm(b), atol). Default: tol=1E-05
                          atol=tol
      num_krylov_vectors: Size of the Krylov space to build at each restart.
        Expense is cubic in this parameter. If supplied, it must be
        an integer in 0 < num_krylov_vectors <= b.size. 
        Default: min(100, b.size).
      maxiter: The Krylov space will be repeatedly rebuilt up to this many
        times. Large values of this argument
        should be used only with caution, since especially for nearly
        symmetric matrices and small `num_krylov_vectors` convergence
        might well freeze at a value significantly larger than `tol`.
        Default: 1
      M: Inverse of the preconditioner of A; see the docstring for
        `scipy.sparse.linalg.gmres`. This is only supported in the
        numpy backend. Supplying this argument to other backends will
        trigger NotImplementedError. Default: None.
      enable_caching: If `True`, block-data during calls to `matvec` is cached
        for later reuse. Note: usually it is safe to enable_caching, unless 
        `matvec` uses matrix decompositions like SVD, QR, eigh, eig or similar.
        In this case, if one does a large number of krylov steps, this can lead 
        to memory clutter and/or OOM errors.
    Raises:
      ValueError: -if `x0` is supplied but its shape differs from that of `b`.
                  -if the ARPACK solver reports a breakdown (which usually 
                   indicates some kind of floating point issue).
                  -if num_krylov_vectors is 0 or exceeds b.size.
                  -if tol was negative.
      TypeError:  -if the dtype of `x0` and `b` are mismatching.

    Returns:
      x: The converged solution. It has the same shape as `b`.
      info: 0 if convergence was achieved, the number of restarts otherwise.
    """

    if x0 is None:
      x0 = self.bs.randn_like(b)

    if not self.bs.compare_shapes(x0, b):
      errstring = (f"x0.sparse_shape = \n{x0.sparse_shape} \ndoes not match "
                   f"b.sparse_shape = \n{b.sparse_shape}.")
      raise ValueError(errstring)

    if x0.dtype != b.dtype:
      raise TypeError(f"x0.dtype = {x0.dtype} does not"
                      f" match b.dtype = {b.dtype}")

    if num_krylov_vectors is None:
      num_krylov_vectors = min(b.size, 100)

    if num_krylov_vectors <= 0 or num_krylov_vectors > b.size:
      errstring = (f"num_krylov_vectors must be in "
                   f"0 < {num_krylov_vectors} <= {b.size}.")
      raise ValueError(errstring)
    if tol < 0:
      raise ValueError(f"tol = {tol} must be positive.")

    if atol is None:
      atol = tol
    elif atol < 0:
      raise ValueError(f"atol = {atol} must be positive.")

    if A_args is None:
      A_args = []
    if A_kwargs is None:
      A_kwargs = {}

    x0.contiguous(inplace=True)
    b.contiguous(inplace=True)
    tmp = BlockSparseTensor(
        numpy.empty(0, dtype=x0.dtype),
        x0._charges,
        x0._flows,
        check_consistency=False)
    def matvec(vector):
      tmp.data = vector
      res = A_mv(tmp, *A_args, **A_kwargs)
      res.contiguous(inplace=True)
      return res.data

    dim = len(x0.data)
    A_op = sp.sparse.linalg.LinearOperator(
        dtype=x0.dtype, shape=(dim, dim), matvec=matvec)

    former_caching_status = self.bs.get_caching_status()
    self.bs.set_caching_status(enable_caching)
    if enable_caching:
      cache_was_empty = self.bs.get_cacher().is_empty
    try:
      x, info = sp.sparse.linalg.gmres(
          A_op,
          b.data,
          x0.data,
          tol=tol,
          atol=atol,
          restart=num_krylov_vectors,
          maxiter=maxiter,
          M=M)
    finally:
      #set caching status back to what it was
      self.bs.set_caching_status(former_caching_status)
      if enable_caching and cache_was_empty:
        self.bs.clear_cache()
    
    if info < 0:
      raise ValueError("ARPACK gmres received illegal input or broke down.")
    if info > 0:
      warnings.warn("gmres did not converge.")
    tmp.data = x
    return tmp, info

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

  def abs(self, tensor: Tensor) -> Tensor:
    return self.bs.abs(tensor)

  def sign(self, tensor: Tensor) -> Tensor:
    return self.bs.sign(tensor)

  def pivot(self, tensor: Tensor, pivot_axis: int = -1) -> Tensor:
    raise NotImplementedError("Symmetric backend doesn't support pivot.")

  def item(self, tensor):
    return tensor.item()

  def matmul(self, tensor1: Tensor, tensor2: Tensor):
    if (tensor1.ndim != 2) or (tensor2.ndim != 2):
      raise ValueError("inputs to `matmul` have to be matrices")
    return tensor1 @ tensor2

  def eps(self, dtype: Type[numpy.number]) -> float:
    """
    Return machine epsilon for given `dtype`

    Args:
      dtype: A dtype.

    Returns:
      float: Machine epsilon.
    """
    return numpy.finfo(dtype).eps
