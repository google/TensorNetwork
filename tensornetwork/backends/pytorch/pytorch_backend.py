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
# pylint: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Callable, List, Type
from typing import Union
from tensornetwork.backends import abstract_backend
from tensornetwork.backends.pytorch import decompositions
import numpy as np

# This might seem bad, but pytype treats tf.Tensor as Any anyway, so
# we don't actually lose anything by doing this.
Tensor = Any

# pylint: disable=abstract-method


class PyTorchBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    super().__init__()
    # pylint: disable=global-variable-undefined
    global torchlib
    try:
      # pylint: disable=import-outside-toplevel
      import torch
    except ImportError as err:
      raise ImportError("PyTorch not installed, please switch to a different "
                        "backend or install PyTorch.") from err
    torchlib = torch
    self.name = "pytorch"

  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    return torchlib.tensordot(a, b, dims=axes)

  def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
    return torchlib.reshape(tensor, tuple(np.array(shape).astype(int)))

  def transpose(self, tensor, perm=None) -> Tensor:
    if perm is None:
      perm = tuple(range(tensor.ndim - 1, -1, -1))
    return tensor.permute(perm)

  def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
            slice_sizes: Tuple[int, ...]) -> Tensor:
    if len(start_indices) != len(slice_sizes):
      raise ValueError("Lengths of start_indices and slice_sizes must be"
                       "identical.")
    obj = tuple(
        slice(start, start + size)
        for start, size in zip(start_indices, slice_sizes))
    return tensor[obj]

  def svd(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      max_singular_values: Optional[int] = None,
      max_truncation_error: Optional[float] = None,
      relative: Optional[bool] = False
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return decompositions.svd(
        torchlib,
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
    return decompositions.qr(torchlib, tensor, pivot_axis, non_negative_diagonal)

  def rq(
      self,
      tensor: Tensor,
      pivot_axis: int = -1,
      non_negative_diagonal: bool = False
  ) -> Tuple[Tensor, Tensor]:
    return decompositions.rq(torchlib, tensor, pivot_axis, non_negative_diagonal)

  def shape_concat(self, values: Tensor, axis: int) -> Tensor:
    return np.concatenate(values, axis)

  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return torchlib.tensor(list(tensor.shape))

  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape)

  def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return self.shape_tuple(tensor)

  def shape_prod(self, values: Tensor) -> int:
    return np.prod(np.array(values))

  def sqrt(self, tensor: Tensor) -> Tensor:
    return torchlib.sqrt(tensor)

  def convert_to_tensor(self, tensor: Tensor) -> Tensor:
    result = torchlib.as_tensor(tensor)
    return result

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return torchlib.tensordot(tensor1, tensor2, dims=0)
  # pylint: disable=unused-argument
  def einsum(self,
             expression: str,
             *tensors: Tensor,
             optimize: bool = True) -> Tensor:
    return torchlib.einsum(expression, *tensors)

  def norm(self, tensor: Tensor) -> Tensor:
    return torchlib.norm(tensor)

  def eye(self,
          N: int,
          dtype: Optional[Any] = None,
          M: Optional[int] = None) -> Tensor:
    dtype = dtype if dtype is not None else torchlib.float64
    if not M:
      M = N  #torch crashes if one passes M = None with dtype!=None
    return torchlib.eye(n=N, m=M, dtype=dtype)

  def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Tensor:
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.ones(shape, dtype=dtype)

  def zeros(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None) -> Tensor:
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.zeros(shape, dtype=dtype)

  def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None,
            seed: Optional[int] = None) -> Tensor:
    if seed:
      torchlib.manual_seed(seed)
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.randn(shape, dtype=dtype)

  def random_uniform(self,
                     shape: Tuple[int, ...],
                     boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                     dtype: Optional[Any] = None,
                     seed: Optional[int] = None) -> Tensor:
    if seed:
      torchlib.manual_seed(seed)
    dtype = dtype if dtype is not None else torchlib.float64
    return torchlib.empty(shape, dtype=dtype).uniform_(*boundaries)

  def conj(self, tensor: Tensor) -> Tensor:
    return tensor  #pytorch does not support complex dtypes

  def eigh(self, matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return matrix.symeig(eigenvectors=True)

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
    of a `LinearOperator` `A`.
    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(vector, *args)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `torch.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If both no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalus. Uses
        `torch.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
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
    if args is None:
      args = []
    #TODO: make this work for tensorflow in graph mode
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

    if not isinstance(initial_state, torchlib.Tensor):
      raise TypeError("Expected a `torch.Tensor`. Got {}".format(
          type(initial_state)))

    initial_state = self.convert_to_tensor(initial_state)
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
      norm_vector_n = torchlib.norm(vector_n)
      if abs(norm_vector_n) < delta:
        break
      norms_vector_n.append(norm_vector_n)
      vector_n = vector_n / norms_vector_n[-1]
      #store the Lanczos vector for later
      if reorthogonalize:
        for v in krylov_vecs:
          vector_n -= (v.contiguous().view(-1).dot(
              vector_n.contiguous().view(-1))) * torchlib.reshape(
                  v, vector_n.shape)
      krylov_vecs.append(vector_n)
      A_vector_n = A(vector_n, *args)
      diag_elements.append(vector_n.contiguous().view(-1).dot(
          A_vector_n.contiguous().view(-1)))

      if ((it > 0) and (it % ndiag) == 0) and (len(diag_elements) >= numeig):
        #diagonalize the effective Hamiltonian
        A_tridiag = torchlib.diag(
            torchlib.tensor(diag_elements)) + torchlib.diag(
                torchlib.tensor(norms_vector_n[1:]), 1) + torchlib.diag(
                    torchlib.tensor(norms_vector_n[1:]), -1)
        eigvals, u = A_tridiag.symeig(eigenvectors=True)
        if not first:
          if torchlib.norm(eigvals[0:numeig] - eigvalsold[0:numeig]) < tol:
            break
        first = False
        eigvalsold = eigvals[0:numeig]
      if it > 0:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
        A_vector_n -= (krylov_vecs[-2] * norms_vector_n[-1])
      else:
        A_vector_n -= (krylov_vecs[-1] * diag_elements[-1])
      vector_n = A_vector_n

    A_tridiag = torchlib.diag(torchlib.tensor(diag_elements)) + torchlib.diag(
        torchlib.tensor(norms_vector_n[1:]), 1) + torchlib.diag(
            torchlib.tensor(norms_vector_n[1:]), -1)
    eigvals, u = A_tridiag.symeig(eigenvectors=True)
    eigenvectors = []
    for n2 in range(min(numeig, len(eigvals))):
      state = self.zeros(initial_state.shape, initial_state.dtype)
      for n1, vec in enumerate(krylov_vecs):
        state += vec * u[n1, n2]
      eigenvectors.append(state / torchlib.norm(state))
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
    #make a copy
    t = torchlib.as_tensor(tensor).clone()
    t[mask] = assignee
    return t

  def inv(self, matrix: Tensor) -> Tensor:
    if len(matrix.shape) > 2:
      raise ValueError(
          "input to pytorch backend method `inv` has shape {}. Only matrices are supported."
          .format(matrix.shape))
    return matrix.inverse()

  def broadcast_right_multiplication(self, tensor1: Tensor,
                                     tensor2: Tensor) -> Tensor:
    if len(tensor2.shape) != 1:
      raise ValueError(
          "only order-1 tensors are allowed for `tensor2`, found `tensor2.shape = {}`"
          .format(tensor2.shape))

    return tensor1 * tensor2

  def broadcast_left_multiplication(self, tensor1: Tensor,
                                    tensor2: Tensor) -> Tensor:
    if len(tensor1.shape) != 1:
      raise ValueError("only order-1 tensors are allowed for `tensor1`,"
                       " found `tensor1.shape = {}`".format(tensor1.shape))

    t1_broadcast_shape = self.shape_concat(
        [self.shape_tensor(tensor1), [1] * (len(tensor2.shape) - 1)], axis=-1)
    return tensor2 * self.reshape(tensor1, t1_broadcast_shape)

  def jit(self, fun: Callable, *args: List, **kwargs: dict) -> Callable:
    return fun

  def sum(self,
          tensor: Tensor,
          axis: Optional[Sequence[int]] = None,
          keepdims: bool = False) -> Tensor:
    return torchlib.sum(tensor, axis=axis, keepdim=keepdims)

  def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    if (tensor1.ndim <= 1) or (tensor2.ndim <= 1):
      raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")

    return torchlib.einsum('...ab,...bc->...ac', tensor1, tensor2)

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
                    Defaults to second-last and last axis (note this
                    differs from the NumPy defaults).
    Returns:
      array_of_diagonals: A dim = min(1, tensor.ndim - 2) tensor storing
                          the batched diagonals.
    """
    if axis1 == axis2:
      raise ValueError("axis1={axis1} and axis2={axis2} must be different.")
    return torchlib.diagonal(tensor, offset=offset, dim1=axis1, dim2=axis2)

  def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
    """ Flattens tensor and creates a new matrix of zeros with its elements
    on the k'th diagonal.
    Args:
      tensor: A tensor.
      k     : The diagonal upon which to place its elements.
    Returns:
      tensor: A new tensor with all zeros save the specified diagonal.
    """
    return torchlib.diag_embed(tensor, offset=k)

  def trace(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
    """Return summed entries along diagonals.

    If tensor is 2-D, the sum is over the
    diagonal of tensor with the given offset,
    i.e., the collection of elements of the form a[i, i+offset].
    If a has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose diagonal is
    summed.

    In the PyTorch backend the trace is always over the main diagonal of the
    last two entries.

    Args:
      tensor: A tensor.
      offset: Offset of the diagonal from the main diagonal.
              This argument is not supported  by the PyTorch
              backend and an error will be raised if they are
              specified.
      axis1, axis2: Axis to be used as the first/second axis of the 2D
                    sub-arrays from which the diagonals should be taken.
                    Defaults to first/second axis.
                    These arguments are not supported by the PyTorch
                    backend and an error will be raised if they are
                    specified.
    Returns:
      array_of_diagonals: The batched summed diagonals.
    """
    if offset != 0:
      errstr = (f"offset = {offset} must be 0 (the default)"
                f"with PyTorch backend.")
      raise NotImplementedError(errstr)
    if axis1 == axis2:
      raise ValueError(f"axis1 = {axis1} cannot equal axis2 = {axis2}")
    N = len(tensor.shape)
    if N > 25:
      raise ValueError(f"Currently only tensors with ndim <= 25 can be traced"
                       f"in the PyTorch backend (yours was {N})")

    if axis1 < 0:
      axis1 = N+axis1
    if axis2 < 0:
      axis2 = N+axis2

    inds = list(map(chr, range(98, 98+N)))
    indsout = [i for n, i in enumerate(inds) if n not in (axis1, axis2)]
    inds[axis1] = 'a'
    inds[axis2] = 'a'
    return torchlib.einsum(''.join(inds) + '->' +''.join(indsout), tensor)

  def abs(self, tensor: Tensor) -> Tensor:
    """
    Returns the elementwise absolute value of tensor.
    Args:
      tensor: An input tensor.
    Returns:
      tensor: Its elementwise absolute value.
    """
    return torchlib.abs(tensor)

  def sign(self, tensor: Tensor) -> Tensor:
    """
    Returns an elementwise tensor with entries
    y[i] = 1, 0, -1 where tensor[i] > 0, == 0, and < 0 respectively.

    For complex input the behaviour of this function may depend on the backend.
    The PyTorch version is not implemented in this case.

    Args:
      tensor: The input tensor.
    """
    return torchlib.sign(tensor)

  def item(self, tensor):
    return tensor.item()

  def eps(self, dtype: Type[np.number]) -> float:
    """
    Return machine epsilon for given `dtype`

    Args:
      dtype: A dtype.

    Returns:
      float: Machine epsilon.
    """
    return torchlib.finfo(dtype).eps
