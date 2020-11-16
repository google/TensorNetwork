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
from tensornetwork.backends.numpy import decompositions
import io
import numpy as np
import scipy as sp
import scipy.sparse.linalg
Tensor = Any

int_to_string = np.array(list(map(chr, list(range(65, 91)))))


class NumPyBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    super().__init__()
    self.name = "numpy"

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    # use einsum for scalar-like products, its much faster
    if not isinstance(axes, int):
      if (len(axes[0]) == a.ndim) and (len(axes[1]) == b.ndim):
        if not len(axes[0]) == len(axes[1]):
          raise ValueError("shape-mismatch for sum")
        u, pos1, _ = np.intersect1d(axes[0],
                                    axes[1],
                                    return_indices=True,
                                    assume_unique=True)
        labels = int_to_string[0:len(u)]
        labels_1 = labels[pos1]
        labels_2 = np.array([''] * len(labels_1))

        labels_2[np.array(axes[1])] = labels
        einsum_label = ','.join([''.join(labels_1), ''.join(labels_2)])
        return np.array(np.einsum(einsum_label, a, b, optimize=True))
      return np.tensordot(a, b, axes)
    return np.tensordot(a, b, axes)

  def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
    return np.reshape(tensor, np.asarray(shape).astype(np.int32))

  def transpose(self,
                tensor: Tensor,
                perm: Optional[Sequence] = None) -> Tensor:
    return np.transpose(tensor, perm)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    obj = tuple(
        slice(start, start + size)
        for start, size in zip(start_indices, slice_sizes))
    return tensor[obj]

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tensor.shape

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tensor.shape

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> Tensor:
    return np.prod(values)

  def sqrt(self, tensor: Tensor) -> Tensor:
    return np.sqrt(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    if (not isinstance(tensor, np.ndarray) and not np.isscalar(tensor)):
      raise TypeError("Expected a `np.array` or scalar. Got {}".format(
          type(tensor)))
    result = np.asarray(tensor)
    return result

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return np.tensordot(tensor1, tensor2, 0)

  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
    return np.einsum(expression, *tensors, optimize=optimize)

  def norm(self, tensor: Tensor) -> Tensor:
    return np.linalg.norm(tensor)

  def eye(self,
          N,
          dtype: Optional[np.dtype] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else np.float64

    return np.eye(N, M=M, dtype=dtype)

  def ones(self,
           shape: Tuple[int, ...],
           dtype: Optional[np.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else np.float64
    return np.ones(shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None) -> Tensor:
    dtype = dtype if dtype is not None else np.float64
    return np.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None,
            seed: Optional[int] = None) -> Tensor:

    if seed:
      np.random.seed(seed)
    dtype = dtype if dtype is not None else np.float64
    if ((np.dtype(dtype) is np.dtype(np.complex128)) or
        (np.dtype(dtype) is np.dtype(np.complex64))):
      return np.random.randn(
          *shape).astype(dtype) + 1j * np.random.randn(*shape).astype(dtype)
    return np.random.randn(*shape).astype(dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[np.dtype] = None,
                     seed: Optional[int] = None) -> Tensor:

    if seed:
      np.random.seed(seed)
    dtype = dtype if dtype is not None else np.float64
    if ((np.dtype(dtype) is np.dtype(np.complex128)) or
        (np.dtype(dtype) is np.dtype(np.complex64))):
      return np.random.uniform(
          boundaries[0],
          boundaries[1], shape).astype(dtype) + 1j * np.random.uniform(
              boundaries[0], boundaries[1], shape).astype(dtype)
    return np.random.uniform(boundaries[0], boundaries[1], shape).astype(dtype)

  def conj(self, tensor: Tensor) -> Tensor:
    return np.conj(tensor)

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return np.linalg.eigh(matrix)

  def eigsh(
      self,
      A: Callable,
      args: Optional[List[Tensor]] = None,
      initial_state: Optional[Tensor] = None,
      shape: Optional[Tuple[int, ...]] = None,
      dtype: Optional[Type[np.number]] = None,  # pylint: disable=no-member
      num_krylov_vecs: int = 50,
      numeig: int = 1,
      tol: float = 1E-8,
      which: Text = 'LR',
      maxiter: Optional[int] = None) -> Tuple[Tensor, List]:
    """Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a symmetric (hermitian) linear operator `A`. `A` is a callable 
    implementing the matrix-vector product. If no `initial_state` is provided 
    then `shape` and `dtype` have to be passed so that a suitable initial
    state can be randomly  generated.
    Args:
      A: A (sparse) implementation of a linear operator
      arsg: A list of arguments to `A`.  `A` will be called as
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
      which : ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI']
        Which `k` eigenvectors and eigenvalues to find:
            'LM' : largest magnitude
            'SM' : smallest magnitude
            'LR' : largest real part
            'SR' : smallest real part
            'LI' : largest imaginary part
            'SI' : smallest imaginary part
        Note that not all of those might be supported by specialized backends.
      maxiter: The maximum number of iterations.
    Returns:
       `Tensor`: An array of `numeig` lowest eigenvalues
       `list`: A list of `numeig` lowest eigenvectors
    """
    raise NotImplementedError("Backend '{}' has not implemented eigs.".format(
        self.name))
  
  def eigs(self,
           A: Callable,
           args: Optional[List] = None,
           initial_state: Optional[Tensor] = None,
           shape: Optional[Tuple[int, ...]] = None,
           dtype: Optional[Type[np.number]] = None,
           num_krylov_vecs: int = 50,
           numeig: int = 6,
           tol: float = 1E-8,
           which: Text = 'LR',
           maxiter: Optional[int] = None) -> Tuple[Tensor, List]:
    """
    Arnoldi method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`. If no `initial_state` is provided then
    `shape` and `dtype` are required so that a suitable initial state can be
    randomly generated.
    This is a wrapper for scipy.sparse.linalg.eigs which only supports
    a subset of the arguments of scipy.sparse.linalg.eigs.

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
    Returns:
       `np.ndarray`: An array of `numeig` lowest eigenvalues
       `list`: A list of `numeig` lowest eigenvectors
    """
    if args is None:
      args = []
    if which in ('SI', 'LI'):
      raise ValueError(f'which = {which} is currently not supported.')

    if numeig + 1 >= num_krylov_vecs:
      raise ValueError('`num_krylov_vecs` > `numeig + 1` required!')

    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, np.ndarray):
      raise TypeError("Expected a `np.ndarray`. Got {}".format(
          type(initial_state)))

    shape = initial_state.shape

    def matvec(vector):
      return np.ravel(A(np.reshape(vector, shape), *args))

    #initial_state is an np.ndarray of rank 1, so we can
    #savely deduce the shape from it
    lop = scipy.sparse.linalg.LinearOperator(dtype=initial_state.dtype,
                                             shape=(initial_state.size,
                                                    initial_state.size),
                                             matvec=matvec)
    eta, U = scipy.sparse.linalg.eigs(A=lop,
                                      k=numeig,
                                      which=which,
                                      v0=initial_state,
                                      ncv=num_krylov_vecs,
                                      tol=tol,
                                      maxiter=maxiter)
    eVs = [np.reshape(U[:, n], shape) for n in range(numeig)]
    return eta, eVs

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

    The numpy backend version of GMRES is simply an interface to
    `scipy.sparse.linalg.gmres`, itself an interace to ARPACK.
    SciPy 1.1.0 or newer (May 05 2018) is required.

    Args:
      A_mv     : A function `v0 = A_mv(v, *A_args, **A_kwargs)` where `v0` and
                 `v` have the same shape.
      b        : The `b` in `A @ x = b`; it should be of the shape `A_mv`
                 operates on.
      A_args   : Positional arguments to `A_mv`, supplied to this interface
                 as a list.
                 Default: None.
      A_kwargs : Keyword arguments to `A_mv`, supplied to this interface
                 as a dictionary.
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
                 Expense is cubic in this parameter. It must be postive.
                 If greater than b.size, it will be set to b.size.
                 Default: 20.
      maxiter  : The Krylov space will be repeatedly rebuilt up to this many
                 times. Large values of this argument
                 should be used only with caution, since especially for nearly
                 symmetric matrices and small `num_krylov_vectors` convergence
                 might well freeze at a value significantly larger than `tol`.
                 Default: 1
      M        : Inverse of the preconditioner of A; see the docstring for
                 `scipy.sparse.linalg.gmres`. This is only supported in the
                 numpy backend. Supplying this argument to other backends will
                 trigger NotImplementedError.
                 Default: None.

    Raises:
      ValueError: -if `x0` is supplied but its shape differs from that of `b`.
                  -if the ARPACK solver reports a breakdown (which usually
                   indicates some kind of floating point issue).
                  -if num_krylov_vectors is 0 or exceeds b.size.
                  -if tol was negative.
      TypeError:  -if the dtype of `x0` and `b` are mismatching.

    Returns:
      x       : The converged solution. It has the same shape as `b`.
      info    : 0 if convergence was achieved, the number of restarts otherwise.
    """
    def matvec(v):
      v_tensor = v.reshape(b.shape)
      Av = A_mv(v_tensor, *A_args, **A_kwargs)
      Avec = Av.ravel()
      return Avec

    A_shape = (b.size, b.size)
    A_op = sp.sparse.linalg.LinearOperator(matvec=matvec,
                                           shape=A_shape,
                                           dtype=b.dtype)
    x, info = sp.sparse.linalg.gmres(A_op,
                                     b.ravel(),
                                     x0,
                                     tol=tol,
                                     atol=atol,
                                     restart=num_krylov_vectors,
                                     maxiter=maxiter,
                                     M=M)
    if info < 0:
      raise ValueError("ARPACK gmres received illegal input or broke down.")
    x = x.reshape(b.shape).astype(b.dtype)
    return (x, info)

  def eigsh_lanczos(self,
                    A: Callable,
                    args: Optional[List[Tensor]] = None,
                    initial_state: Optional[Tensor] = None,
                    shape: Optional[Tuple] = None,
                    dtype: Optional[Type[np.number]] = None,
                    num_krylov_vecs: int = 20,
                    numeig: int = 1,
                    tol: float = 1E-8,
                    delta: float = 1E-8,
                    ndiag: int = 20,
                    reorthogonalize: bool = False) -> Tuple[Tensor, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`.
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

    if not isinstance(initial_state, np.ndarray):
      raise TypeError("Expected a `np.ndarray`. Got {}".format(
          type(initial_state)))

    vector_n = initial_state
    Z = self.norm(vector_n)
    vector_n /= Z
    norms_vector_n = []
    diag_elements = []
    krylov_vecs = []
    first = True
    eigvalsold = []
    for it in range(num_krylov_vecs):
      #normalize the current vector:
      norm_vector_n = self.norm(vector_n)
      if abs(norm_vector_n) < delta:
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      #store the Lanczos vector for later
      if reorthogonalize:
        for v in krylov_vecs:
          vector_n -= np.dot(np.ravel(np.conj(v)), np.ravel(vector_n)) * v
      krylov_vecs.append(vector_n)
      A_vector_n = A(vector_n, *args)
      diag_elements.append(
          np.dot(np.ravel(np.conj(vector_n)), np.ravel(A_vector_n)))

      if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
        #diagonalize the effective Hamiltonian
        A_tridiag = np.diag(diag_elements) + np.diag(
            norms_vector_n[1:], 1) + np.diag(np.conj(norms_vector_n[1:]), -1)
        eigvals, u = np.linalg.eigh(A_tridiag)
        if not first:
          if np.linalg.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = np.diag(diag_elements) + np.diag(
        norms_vector_n[1:], 1) + np.diag(np.conj(norms_vector_n[1:]), -1)
    eigvals, u = np.linalg.eigh(A_tridiag)
    eigenvectors = []
    eigvals = np.array(eigvals).astype(A_tridiag.dtype)

    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / np.linalg.norm(state))
    return eigvals[0:numeig], eigenvectors

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
    t = np.copy(tensor)
    t[mask] = assignee
    return t

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError("input to numpy backend method `inv` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    return np.linalg.inv(matrix)

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
    return np.sin(tensor)

  def cos(self, tensor: Tensor) -> Tensor:
    return np.cos(tensor)

  def exp(self, tensor: Tensor) -> Tensor:
    return np.exp(tensor)

  def log(self, tensor: Tensor) -> Tensor:
    return np.log(tensor)

  def expm(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) != 2:
      raise ValueError("input to numpy backend method `expm` has shape {}."
                       " Only matrices are supported.".format(matrix.shape))
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("input to numpy backend method `expm` only supports"
                       " N*N matrix, {x}*{y} matrix is given".format(
                           x=matrix.shape[0], y=matrix.shape[1]))
    # pylint: disable=no-member
    return sp.linalg.expm(matrix)

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    return fun

  def sum(self,
          tensor: Tensor,
          axis: Optional[Sequence[int]] = None,
          keepdims: bool = False) -> Tensor:
    return np.sum(tensor, axis=tuple(axis), keepdims=keepdims)

  def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if (tensor1.ndim <= 1) or (tensor2.ndim <= 1):
      raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")
    return np.matmul(tensor1, tensor2)

  def svd(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd(np,
                              tensor,
                              pivot_axis,
                              max_singular_values,
                              max_truncation_error,
                              relative=relative)

  def qr(self,
         tensor: Tensor,
         pivot_axis: int = -1,
         non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
    #pylint: disable=too-many-function-args
    return decompositions.qr(np, tensor, pivot_axis, non_negative_diagonal)

  def rq(self,
         tensor: Tensor,
         pivot_axis: int = -1,
         non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
    #pylint: disable=too-many-function-args
    return decompositions.rq(np, tensor, pivot_axis, non_negative_diagonal)

  def diagonal(self,
               tensor: Tensor,
               offset: int = 0,
               axis1: int = -2,
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
                    Defaults to second-last/last axis.
    Returns:
      array_of_diagonals: A dim = min(1, tensor.ndim - 2) tensor storing
                          the batched diagonals.
    """
    return np.diagonal(tensor, offset=offset, axis1=axis1, axis2=axis2)

  def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
    """ Flattens tensor and creates a new matrix of zeros with its elements
    on the k'th diagonal.
    Args:
      tensor: A tensor.
      k     : The diagonal upon which to place its elements.
    Returns:
      tensor: A new tensor with all zeros save the specified diagonal.
    """
    return np.diagflat(tensor, k=k)

  def trace(self,
            tensor: Tensor,
            offset: int = 0,
            axis1: int = -2,
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
                    Defaults to second-last/last axis.
    Returns:
      array_of_diagonals: The batched summed diagonals.
    """
    return np.trace(tensor, offset=offset, axis1=axis1, axis2=axis2)

  def abs(self, tensor: Tensor) -> Tensor:
    """
    Returns the elementwise absolute value of tensor.
    Args:
      tensor: An input tensor.
    Returns:
      tensor: Its elementwise absolute value.
    """
    return np.abs(tensor)

  def sign(self, tensor: Tensor) -> Tensor:
    """
    Returns an elementwise tensor with entries
    y[i] = 1, 0, -1 tensor[i] > 0, == 0, and < 0 respectively.

    For complex input the behaviour of this function may depend on the backend.
    The NumPy version returns y[i] = x[i]/sqrt(x[i]^2).

    Args:
      tensor: The input tensor.
    """
    return np.sign(tensor)

  def serialize_tensor(self, tensor: Tensor) -> str:
    """
    Return a string that serializes the given tensor.
    
    Args:
      tensor: The input tensor.
      
    Returns:
      A string representing the serialized tensor. 
    """
    m = io.BytesIO()
    np.save(m, tensor, allow_pickle=False)
    m.seek(0)
    return str(m.read(), encoding='latin-1')

  def deserialize_tensor(self, s: str) -> Tensor:
    """
    Return a tensor given a serialized tensor string. 
    
    Args:
      s: The input string representing a serialized tensor.
      
    Returns:
      The tensor object represented by the string.
     
    """
    m = io.BytesIO()
    m.write(s.encode('latin-1'))
    m.seek(0)
    return np.load(m)
  
  def power(self, a: Tensor, b: Union[Tensor, float]) -> Tensor:
    """
    Returns the exponentiation of tensor a raised to b.  
      If b is a tensor, then the exponentiation is element-wise 
        between the two tensors, with a as the base and b as the power.
        Note that a and b must be broadcastable to the same shape if 
        b is a tensor.
      If b is a scalar, then the exponentiation is each value in a
        raised to the power of b.
    
    Args:
      a: The tensor containing the bases.
      b: The tensor containing the powers; or a single scalar as the power.

    Returns:
      The tensor that is each element of a raised to the 
        power of b.  Note that the shape of the returned tensor
        is that produced by the broadcast of a and b.
    """
    return np.power(a, b)
  
  def item(self, tensor):
    return tensor.item()
