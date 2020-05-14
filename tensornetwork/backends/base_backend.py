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
#pylint: disable=line-too-long
from typing import Optional, Sequence, Tuple, Any, Union, Type, Callable, List, Text
import numpy as np
# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
Tensor = Any


class BaseBackend:

  def __init__(self):
    self.name = 'base backend'

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Sequence[Sequence[int]]) -> Tensor:
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

  def transpose(self, tensor: Tensor, perm: Sequence[int]) -> Tensor:
    """Transpose a tensor according to a given permutation
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

  def svd_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the singular value decomposition (SVD) of a tensor.

    The SVD is performed by treating the tensor as a matrix, with an effective
    left (row) index resulting from combining the axes
    `tensor.shape[:split_axis]` and an effective right (column) index resulting
    from combining the axes `tensor.shape[split_axis:]`.

    For example, if `tensor` had a shape (2, 3, 4, 5) and `split_axis` was 2,
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
      split_axis: Where to split the tensor's axes before flattening into a
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
    raise NotImplementedError(
        "Backend '{}' has not implemented svd_decomposition.".format(self.name))

  def qr_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a tensor."""
    raise NotImplementedError(
        "Backend '{}' has not implemented qr_decomposition.".format(self.name))

  def rq_decomposition(
      self,
      tensor: Tensor,
      split_axis: int,
  ) -> Tuple[Tensor, Tensor]:
    """Computes the RQ (reversed QR) decomposition of a tensor."""
    raise NotImplementedError(
        "Backend '{}' has not implemented rq_decomposition.".format(self.name))

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

  def diag(self, tensor: Tensor) -> Tensor:
    """Create a diagonal matrix from the given vector tensor."""
    raise NotImplementedError("Backend '{}' has not implemented diag.".format(
        self.name))

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    """Convert a np.array or a tensor to a tensor type for the backend."""
    raise NotImplementedError(
        "Backend '{}' has not implemented convert_to_tensor.".format(self.name))

  def trace(self, tensor: Tensor) -> Tensor:
    """Calculate the trace over the last two axes of the given tensor."""
    raise NotImplementedError("Backend '{}' has not implemented trace.".format(
        self.name))

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Calculate the outer product of the two given tensors."""
    raise NotImplementedError(
        "Backend '{}' has not implemented outer_product.".format(self.name))

  def einsum(self, expression: str, *tensors: Tensor) -> Tensor:
    """Calculate sum of products of tensors according to expression."""
    raise NotImplementedError("Backend '{}' has not implemented einsum.".format(
        self.name))

  def norm(self, tensor: Tensor) -> Tensor:
    """Calculate the L2-norm of the elements of `tensor`
    """
    raise NotImplementedError("Backend '{}' has not implemented norm.".format(
        self.name))

  def eye(self,
          N: int,
          dtype: Type[np.number],
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

  def ones(self, shape: Tuple[int, ...], dtype: Type[np.number]) -> Tensor:
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

  def zeros(self, shape: Tuple[int, ...], dtype: Type[np.number]) -> Tensor:
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

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Type[np.number]] = None,
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

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[Type[np.number]] = None,
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

  def eigs(self,
           A: Callable,
           args: Optional[List[Tensor]] = None,
           initial_state: Optional[Tensor] = None,
           shape: Optional[Tuple[int, ...]] = None,
           dtype: Optional[Type[np.number]] = None,
           num_krylov_vecs: int = 50,
           numeig: int = 1,
           tol: float = 1E-8,
           which: Text = 'LR',
           maxiter: Optional[int] = None) -> List[Tensor]:
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
      maxiter: The maximum number of iterations.
    Returns:
       `Tensor`: An array of `numeig` lowest eigenvalues
       `Tensor`: An array of `numeig` lowest eigenvectors
    """
    raise NotImplementedError("Backend '{}' has not implemented eigs.".format(
        self.name))

  def eigsh_lanczos(self,
                    A: Callable,
                    args: Optional[List[Tensor]] = None,
                    initial_state: Optional[Tensor] = None,
                    shape: Optional[Tuple[int, ...]] = None,
                    dtype: Optional[Type[np.number]] = None,
                    num_krylov_vecs: int = 20,
                    numeig: int = 1,
                    tol: float = 1E-8,
                    delta: float = 1E-8,
                    ndiag: int = 20,
                    reorthogonalize: bool = False) -> Tuple[List, List]:
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
    `tensor1` * tensor2`, where `tensor1` is an arbitrary tensor and `tensor2` is a
    one-dimensional tensor. The broadcasting is applied to the last index of
    `tensor1`.
    Args:
      tensor1: A tensor.
      tensor2: A tensor.
    Returns:
      Tensor: The result of multiplying `tensor1` onto `tensor2`.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented `broadcast_right_multiplication`."
        .format(self.name))

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    """
    Perform broadcasting for multiplication of `tensor1` onto `tensor2`, i.e.
    `tensor1` * tensor2`, where `tensor2` is an arbitrary tensor and `tensor1` is a
    one-dimensional tensor. The broadcasting is applied to the first index of
    `tensor2`.
    Args:
      tensor1: A tensor.
      tensor2: A tensor.
    Returns:
      Tensor: The result of multiplying `tensor1` onto `tensor2`.
    """
    raise NotImplementedError(
        "Backend '{}' has not implemented `broadcast_left_multiplication`."
        .format(self.name))

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

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    """
    Return a jitted or graph-compiled version of `fun` \
    for jax and tensorflow backends. For all other backends
    returns `fun`.
    Args:
      fun: Callable
      args: Arguments to `fun`.
      kwargs: Keyword arguments to `fun`.  
    Returns:
      Callable: jitted/graph-compiled version of `fun`, or just `fun`.
    """
    raise NotImplementedError("Backend '{}' has not implemented `jit`.".format(
        self.name))
