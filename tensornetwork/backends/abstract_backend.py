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
from typing import (Optional, Sequence, Tuple, Any, Union, Type, Callable, List,
                    Text)
import numpy as np
# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
Tensor = Any


class AbstractBackend:

  def __init__(self) -> None:
    self.name = 'abstract backend'

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    """Do a tensordot of tensors `a` and `b` over the given axes.

    Args:
      a: A tensor.
      b: Another tensor.
      axes: Two lists of integers. These values are the contraction
        axes.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented tensordot.".format(self.name))

  # We use `Tensor` for the shape type here since the shape could
  # be a tensor.
  def reshape(self, tensor: Tensor, shape: Sequence[Tensor]) -> Tensor:
    """Reshape tensor to the given shape.

    Args:
      tensor: A tensor.
    Returns:
      The reshaped tensor.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented reshape.".format(self.name))

  def transpose(self,
                tensor: Tensor,
                perm: Optional[Sequence[int]] = None) -> Tensor:
    """Transpose a tensor according to a given permutation. By default
    the axes are reversed.
    Args:
      tensor: A tensor.
      perm: The permutation of the axes.
    Returns:
      The transposed tensor
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented transpose.".format(self.name))

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    """Obtains a slice of a tensor based on start_indices and slice_sizes.

    Args:
      tensor: A tensor.
      start_indices: Tuple of integers denoting start indices of slice.
      slice_sizes: Tuple of integers denoting size of slice along each axis.
    """
    raise NotImplementedError("Backend '{}' has not implemented slice.".format(
        self.name))

  def svd(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the singular value decomposition (SVD) of a tensor.

    The SVD is performed by treating the tensor as a matrix, with an effective
    left (row) index resulting from combining the axes
    `tensor.shape[:pivot_axis]` and an effective right (column) index resulting
    from combining the axes `tensor.shape[pivot_axis:]`.

    For example, if `tensor` had a shape (2, 3, 4, 5) and `pivot_axis` was 2,
    then `u` would have shape (2, 3, 6), `s` would have shape (6), and `vh`
    would have shape (6, 4, 5).

    If `max_singular_values` is set to an integer, the SVD is truncated to keep
    at most this many singular values.

    If `max_truncation_error > 0`, as many singular values will be truncated as
    possible, so that the truncation error (the norm of discarded singular
    values) is at most `max_truncation_error`.
    If `relative` is set `True` then `max_truncation_err` is understood
    relative to the largest singular value.

    If both `max_singular_values` and `max_truncation_error` are specified, the
    number of retained singular values will be
    `min(max_singular_values, nsv_auto_trunc)`, where `nsv_auto_trunc` is the
    number of singular values that must be kept to maintain a truncation error
    smaller than `max_truncation_error`.

    The output consists of three tensors `u, s, vh` such that:
    ```python
      u[i1,...,iN, j] * s[j] * vh[j, k1,...,kM] == tensor[i1,...,iN, k1,...,kM]
    ```
    Note that the output ordering matches numpy.linalg.svd rather than tf.svd.

    Args:
      tensor: A tensor to be decomposed.
      pivot_axis: Where to split the tensor's axes before flattening into a
        matrix.
      max_singular_values: The number of singular values to keep, or `None` to
        keep them all.
      max_truncation_error: The maximum allowed truncation error or `None` to
        not do any truncation.
      relative: Multiply `max_truncation_err` with the largest singular value.

    Returns:
      u: Left tensor factor.
      s: Vector of ordered singular values from largest to smallest.
      vh: Right tensor factor.
      s_rest: Vector of discarded singular values (length zero if no
              truncation).
    """
    raise NotImplementedError("Backend '{}' has not implemented svd.".format(
        self.name))

  def qr(self,
         tensor: Tensor,
         pivot_axis: int = -1,
         non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a tensor."""
    raise NotImplementedError("Backend '{}' has not implemented qr.".format(
        self.name))

  def rq(self,
         tensor: Tensor,
         pivot_axis: int = -1,
         non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
    """Computes the RQ (reversed QR) decomposition of a tensor."""
    raise NotImplementedError("Backend '{}' has not implemented rq.".format(
        self.name))

  def shape_concat(self, values: Sequence[Tensor], axis) -> Tensor:
    """Concatenate a sequence of tensors together about the given axis."""
    raise NotImplementedError("Backend '{}' has not implemented concat.".format(
        self.name))

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    """Get the shape of a tensor.

    Args:
      tensor: A tensor.
    Returns:
      The shape of the input tensor returned as another tensor.
    """
    raise NotImplementedError("Backend '{}' has not implemented shape.".format(
        self.name))

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    """Get the shape of a tensor as a tuple of integers.

    Args:
      tensor: A tensor.

    Returns:
      The shape of the input tensor returned as a tuple of ints.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented shape_tuple.".format(self.name))

  def sparse_shape(self, tensor: Tensor) -> Any:
    raise NotImplementedError(
        "Backend '{}' has not implemented `sparse_shape`.".format(self.name))

  def shape_prod(self, values: Tensor) -> Tensor:
    """Take the product of all of the elements in values"""
    raise NotImplementedError("Backend '{}' has not implemented prod.".format(
        self.name))

  def sqrt(self, tensor: Tensor) -> Tensor:
    """Take the square root (element wise) of a given tensor."""
    raise NotImplementedError("Backend '{}' has not implemented sqrt.".format(
        self.name))

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    """Convert a np.array or a tensor to a tensor type for the backend."""
    raise NotImplementedError(
        "Backend '{}' has not implemented convert_to_tensor.".format(self.name))

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Calculate the outer product of the two given tensors."""
    raise NotImplementedError(
        "Backend '{}' has not implemented outer_product.".format(self.name))

  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
    """Calculate sum of products of tensors according to expression."""
    raise NotImplementedError("Backend '{}' has not implemented einsum.".format(
        self.name))

  def norm(self, tensor: Tensor) -> Tensor:
    """Calculate the L2-norm of the elements of `tensor`
    """
    raise NotImplementedError("Backend '{}' has not implemented norm.".format(
        self.name))

  def eye(
      self,
      N: int,
      dtype: Type[np.number],  # pylint: disable=no-member
      M: Optional[int] = None) -> Tensor:
    """Return an identity matrix of dimension `dim`
       Depending on specific backends, `dim` has to be either an int
       (numpy, torch, tensorflow) or a `ShapeType` object
       (for block-sparse backends). Block-sparse
       behavior is currently not supported
      Args:
        N (int): The dimension of the returned matrix.
        dtype: The dtype of the returned matrix.
        M (int): The dimension of the returned matrix.
    """
    #TODO: implement `ShapeType` objects
    raise NotImplementedError("Backend '{}' has not implemented eye.".format(
        self.name))

  def ones(self, shape: Tuple[int, ...], dtype: Type[np.number]) -> Tensor:  # pylint: disable=no-member
    """Return an ones-matrix of dimension `dim`
       Depending on specific backends, `dim` has to be either an int
       (numpy, torch, tensorflow) or a `ShapeType` object
       (for block-sparse backends). Block-sparse
       behavior is currently not supported
       Args:
         shape (int): The dimension of the returned matrix.
         dtype: The dtype of the returned matrix.
    """
    raise NotImplementedError("Backend '{}' has not implemented ones.".format(
        self.name))

  def zeros(self, shape: Tuple[int, ...], dtype: Type[np.number]) -> Tensor:  # pylint: disable=no-member
    """Return a zeros-matrix of dimension `dim` Depending on specific backends,
    `dim` has to be either an int (numpy, torch, tensorflow) or a `ShapeType`
    object (for block-sparse backends).

    Block-sparse
    behavior is currently not supported
    Args:
      shape (int): The dimension of the returned matrix.
      dtype: The dtype of the returned matrix.
    """
    raise NotImplementedError("Backend '{}' has not implemented zeros.".format(
        self.name))

  def randn(
      self,
      shape: Tuple[int, ...],
      dtype: Optional[Type[np.number]] = None,  # pylint: disable=no-member
      seed: Optional[int] = None) -> Tensor:
    """Return a random-normal-matrix of dimension `dim` Depending on specific
    backends, `dim` has to be either an int (numpy, torch, tensorflow) or a
    `ShapeType` object (for block-sparse backends).

    Block-sparse
    behavior is currently not supported
    Args:
      shape (int): The dimension of the returned matrix.
      dtype: The dtype of the returned matrix.
      seed:  The seed for the random number generator
    """
    raise NotImplementedError("Backend '{}' has not implemented randn.".format(
        self.name))

  def random_uniform(
      self,
      shape: Tuple[int, ...],
      boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
      dtype: Optional[Type[np.number]] = None,  # pylint: disable=no-member
      seed: Optional[int] = None) -> Tensor:
    """Return a random uniform matrix of dimension `dim`.

    Depending on specific backends, `dim` has to be either an int
    (numpy, torch, tensorflow) or a `ShapeType` object
    (for block-sparse backends). Block-sparse
    behavior is currently not supported
    Args:
      shape (int): The dimension of the returned matrix.
      boundaries (tuple): The boundaries of the uniform distribution.
      dtype: The dtype of the returned matrix.
      seed:  The seed for the random number generator
    Returns:
      Tensor : random uniform initialized tensor.
    """
    raise NotImplementedError(("Backend '{}' has not implemented "
                               "random_uniform.").format(self.name))

  def conj(self, tensor: Tensor) -> Tensor:
    """
    Return the complex conjugate of `tensor`
    Args:
      tensor: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented conj.".format(
        self.name))

  def eigh(self, matrix: Tensor):
    """Compute eigenvectors and eigenvalues of a hermitian matrix.

    Args:
      matrix: A symetric matrix.
    Returns:
      Tensor: The eigenvalues in ascending order.
      Tensor: The eigenvectors.
    """
    raise NotImplementedError("Backend '{}' has not implemented eigh".format(
        self.name))

  def eigs(
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
    """Arnoldi method for finding the lowest eigenvector-eigenvalue pairs
    of a linear operator `A`. `A` is a callable implementing the
    matrix-vector product. If no `initial_state` is provided then
    `shape` and `dtype` have to be passed so that a suitable initial
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

  def eigsh_lanczos(self,
                    A: Callable,
                    args: Optional[List[Tensor]] = None,
                    initial_state: Optional[Tensor] = None,
                    shape: Optional[Tuple[int, ...]] = None,
                    dtype: Optional[Type[np.number]] = None,# pylint: disable=no-member
                    num_krylov_vecs: int = 20,
                    numeig: int = 1,
                    tol: float = 1E-8,
                    delta: float = 1E-8,
                    ndiag: int = 20,
                    reorthogonalize: bool = False) -> Tuple[Tensor, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of `A`.
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
        `backend.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
        as stopping criterion between two diagonalization steps of the
        tridiagonal operator.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. It means that an (approximate) invariant subspace has
        been found.
      ndiag: The tridiagonal Operator is diagonalized every `ndiag`
        iterations to check convergence.
      reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
        explicit orthogonalization (more costly than `reorthogonalize=False`)
    Returns:
      (eigvals, eigvecs)
       eigvals: A list of `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented eighs_lanczos.".format(self.name))

  def gmres(self,
            A_mv: Callable,
            b: Tensor,
            A_args: Optional[List] = None,
            A_kwargs: Optional[dict] = None,
            x0: Optional[Tensor] = None,
            tol: float = 1E-05,
            atol: Optional[float] = None,
            num_krylov_vectors: int = 20,
            maxiter: Optional[int] = 1,
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
                 Expense is cubic in this parameter. It must be positive.
                 If greater than b.size, it will be set to b.size.
                 Default: 20
      maxiter  : The Krylov space will be repeatedly rebuilt up to this many
                 times. Large values of this argument
                 should be used only with caution, since especially for nearly
                 symmetric matrices and small `num_krylov_vectors` convergence
                 might well freeze at a value significantly larger than `tol`.
                 Default: 1.
      M        : Inverse of the preconditioner of A; see the docstring for
                 `scipy.sparse.linalg.gmres`. This is only supported in the
                 numpy backend. Supplying this argument to other backends will
                 trigger NotImplementedError.
                 Default: None.

    Raises:
      ValueError: -if `x0` is supplied but its shape differs from that of `b`.
                  -in NumPy, if the ARPACK solver reports a breakdown (which
                   usually indicates some kind of floating point issue).
                  -if num_krylov_vectors is 0 or exceeds b.size.
                  -if tol was negative.
                  -if M was supplied with any backend but NumPy.

    Returns:
      x       : The converged solution. It has the same shape as `b`.
      info    : 0 if convergence was achieved, the number of restarts otherwise.
    """
    bshape = self.shape_tensor(b)
    N = self.shape_prod(bshape)
    try:
      dtype = b.dtype
    except AttributeError as err:
      raise AttributeError("gmres was called using a vector `b` that did"
                           "not have a dtype method.") from err

    if x0 is None:
      x0 = self.zeros((N,), dtype)
    else:
      x0shape = self.shape_tensor(x0)
      if x0shape != bshape:
        errstring = (f"If x0 is supplied, its shape, {x0shape}, must match b's"
                     f", {bshape}.")
        raise ValueError(errstring)
      try:
        x0dtype = x0.dtype
      except AttributeError as err:
        raise AttributeError("gmres was called using a vector `x0` that did"
                             "not have a dtype method.") from err
      if x0dtype != dtype:
        errstring = (f"If x0 is supplied, its dtype, {x0dtype}, must match"
                     f" b's, {dtype}.")
        raise TypeError(errstring)

      x0 = self.reshape(x0, (N,))

    if num_krylov_vectors > N:
      num_krylov_vectors = N

    if tol < 0:
      raise ValueError(f"tol = {tol} must be positive.")

    if atol is None:
      atol = tol
    elif atol < 0:
      raise ValueError(f"atol = {atol} must be positive.")

    if num_krylov_vectors <= 0:
      errstring = (f"num_krylov_vectors must be positive, not"
                   f"{num_krylov_vectors}.")
      raise ValueError(errstring)

    if A_args is None:
      A_args = []
    if A_kwargs is None:
      A_kwargs = {}
    return self._gmres(A_mv, b, A_args, A_kwargs, x0, tol, atol,
                       num_krylov_vectors, maxiter, M=M)


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
    raise NotImplementedError("Backend '{}' has not implemented gmres.".format(
        self.name))


  def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
      Return the default addition of `tensor`.
      A backend can override such implementation.
      Args:
        tensor1: A tensor.
        tensor2: A tensor.
      Returns:
        Tensor
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented addition.".format(self.name))

  def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
      Return the default substraction of `tensor`.
      A backend can override such implementation.
      Args:
        tensor1: A tensor.
        tensor2: A tensor.
      Returns:
        Tensor
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented subtraction.".format(self.name))

  def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Return the default multiplication of `tensor`.

    A backend can override such implementation.
    Args:
      tensor1: A tensor.
      tensor2: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented multiply.".format(self.name))

  def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
      Return the default divide of `tensor`.
      A backend can override such implementation.
      Args:
        tensor1: A tensor.
        tensor2: A tensor.
      Returns:
        Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented divide.".format(
        self.name))

  def index_update(self, tensor: Tensor, mask: Tensor,
                   assignee: Tensor) -> Tensor:
    """Update `tensor` at elements defined by `mask` with value `assignee`.

    Args:
      tensor: A `Tensor` object.
      mask: A boolean mask.
      assignee: A scalar `Tensor`. The values to assigned to `tensor`
        at positions where `mask` is `True`.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented `index_update`.".format(self.name))

  def inv(self, matrix: Tensor) -> Tensor:
    """Compute the matrix inverse of `matrix`.

    Args:
      matrix: A matrix.
    Returns:
      Tensor: The inverse of `matrix`
    """
    raise NotImplementedError("Backend '{}' has not implemented `inv`.".format(
        self.name))

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    """
    Perform broadcasting for multiplication of `tensor2` onto `tensor1`, i.e.
    `tensor1` * tensor2`, where `tensor1` is an arbitrary tensor and `tensor2`
    is a one-dimensional tensor. The broadcasting is applied to the last index
    of `tensor1`.
    Args:
      tensor1: A tensor.
      tensor2: A tensor.
    Returns:
      Tensor: The result of multiplying `tensor1` onto `tensor2`.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented `broadcast_right_multiplication`.".
        format(self.name))

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    """
    Perform broadcasting for multiplication of `tensor1` onto `tensor2`, i.e.
    `tensor1` * tensor2`, where `tensor2` is an arbitrary tensor and `tensor1`
    is a one-dimensional tensor. The broadcasting is applied to the first
    index of `tensor2`.
    Args:
      tensor1: A tensor.
      tensor2: A tensor.
    Returns:
      Tensor: The result of multiplying `tensor1` onto `tensor2`.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented `broadcast_left_multiplication`.".
        format(self.name))

  def sin(self, tensor: Tensor) -> Tensor:
    """
    Return sin of `tensor`.
    Args:
      tensor: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented `sin`.".format(
        self.name))

  def cos(self, tensor: Tensor) -> Tensor:
    """
    Return cos of `tensor`.
    Args:
      tensor: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented `cos`.".format(
        self.name))

  def exp(self, tensor: Tensor) -> Tensor:
    """
    Return elementwise exp of `tensor`.
    Args:
      tensor: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented `exp`.".format(
        self.name))

  def log(self, tensor: Tensor) -> Tensor:
    """
    Return elementwise natural logarithm of `tensor`.
    Args:
      tensor: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented `log`.".format(
        self.name))

  def expm(self, matrix: Tensor) -> Tensor:
    """
    Return expm log of `matrix`, matrix exponential.
    Args:
      matrix: A tensor.
    Returns:
      Tensor
    """
    raise NotImplementedError("Backend '{}' has not implemented `expm`.".format(
        self.name))

  def jit(self, fun: Callable, *args: Any, **kwargs: Any) -> Callable:
    """
    Return a jitted or graph-compiled version of `fun`
    for JAX backend. For all other backends returns `fun`.
    Args:
      fun: Callable
      args: Arguments to `fun`.
      kwargs: Keyword arguments to `fun`.
    Returns:
      Callable: jitted/graph-compiled version of `fun`, or just `fun`.
    """
    raise NotImplementedError("Backend '{}' has not implemented `jit`.".format(
        self.name))

  def sum(self,
          tensor: Tensor,
          axis: Optional[Sequence[int]] = None,
          keepdims: bool = False) -> Tensor:
    """
    Sum elements of `tensor` along the specified `axis`. Results in a
    new Tensor with the summed axis removed.
    Args:
      tensor: An input tensor.
    Returns:
      tensor: The result of performing the summation. The order of the tensor
        will be reduced by 1.
    """
    raise NotImplementedError("Backend '{}' has not implemented `sum`.".format(
        self.name))

  def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Perform a possibly batched matrix-matrix multiplication
    between `tensor1` and `tensor2`. The following behaviour
    is similar to `numpy.matmul`:
    - If both arguments are 2-D they are multiplied like conventional
      matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of
      matrices residing in the last two indexes and broadcast accordingly.
    Both arguments to `matmul` have to be tensors of order >= 2.
    Args:
      tensor1: An input tensor.
      tensor2: An input tensor.
    Returns:
      tensor: The result of performing the matmul.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented `matmul`.".format(self.name))

  def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
    """ Flattens tensor and creates a new matrix of zeros with its elements
    on the k'th diagonal.
    Args:
      tensor: A tensor.
      k     : The diagonal upon which to place its elements.
    Returns:
      tensor: A new tensor with all zeros save the specified diagonal.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented diagflat.".format(self.name))

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
    raise NotImplementedError(
        "Backend '{}' has not implemented diagonal.".format(self.name))

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
    raise NotImplementedError("Backend '{}' has not implemented trace.".format(
        self.name))

  def abs(self, tensor: Tensor) -> Tensor:
    """
    Returns the elementwise absolute value of tensor.
    Args:
      tensor: An input tensor.
    Returns:
      tensor: Its elementwise absolute value.
    """
    raise NotImplementedError("Backend '{}' has not implemented `abs`.".format(
        self.name))

  def sign(self, tensor: Tensor):
    """
    Returns an elementwise tensor with entries
    y[i] = 1, 0, -1 where tensor[i] > 0, == 0, and < 0 respectively.

    Args:
      tensor: The input tensor.
    """
    raise NotImplementedError("Backend '{}' has not implemented `sign`.".format(
        self.name))

  def pivot(self, tensor: Tensor, pivot_axis: int = -1) -> Tensor:
    """ Reshapes a tensor into a matrix, whose columns (rows) are the
    vectorized dimensions to the left (right) of pivot_axis.

    In other words, with tensor.shape = (1, 2, 4, 5) and pivot_axis=2,
    this function returns an (8, 5) matrix.

    Args:
      tensor: The tensor to pivot.
      pivot_axis: The axis about which to pivot.

    Returns:
      The pivoted tensor.
    """
    ndim = len(self.shape_tuple(tensor))
    if pivot_axis > ndim:
      errstr = f"pivot_axis = {pivot_axis} was invalid given ndim={ndim} array."
      raise ValueError(errstr)

    left_dims = tensor.shape[:pivot_axis]
    right_dims = tensor.shape[pivot_axis:]
    tensor = self.reshape(
        tensor, [self.shape_prod(left_dims),
                 self.shape_prod(right_dims)])
    return tensor

  def serialize_tensor(self, tensor: Tensor) -> str:
    """
    Return a string that serializes the given tensor.

    Args:
      tensor: The input tensor.

    Returns:
      A string representing the serialized tensor.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented serialize_tensor.".format(self.name))

  def deserialize_tensor(self, s: str) -> Tensor:
    """
    Return a tensor given a serialized tensor string.

    Args:
      s: The input string representing a serialized tensor.

    Returns:
      The tensor object represented by the string.

    """
    raise NotImplementedError(
        "Backend '{}' has not implemented deserialize_tensor.".format(
            self.name))

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
    raise NotImplementedError(
        f"Backend {self.name} has not implemented power.")

  def item(self, tensor) -> Union[float, int, complex]:
    """
    Return the item of a 1-element tensor.

    Args:
      tensor: A 1-element tensor

    Returns:
      The value in tensor.
    """
    raise NotImplementedError("Backend {self.name} has not implemented item")

  def cholesky(self, 
               tensor: Tensor,
               pivot_axis: int = -1,
               non_negative_diagonal: bool = False) -> \
               Tuple[Tensor, Tensor]:
    raise NotImplementedError(
        f"Backend {self.name} has not implemented cholesky.")

  def eps(self, dtype: Type[np.number]) -> float:
    """
    Return machine epsilon for given `dtype`

    Args:
      dtype: A dtype.

    Returns:
      float: Machine epsilon.
    """

    raise NotImplementedError(
        f"Backend {self.name} has not implemented eps.")
