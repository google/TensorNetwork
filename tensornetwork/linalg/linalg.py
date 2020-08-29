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

from typing import Optional, Tuple
from tensornetwork.tensor import Tensor


def svd(
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
  backend = tensor.backend
  out = backend.svd(tensor.array, pivot_axis,
                    max_singular_values=max_singular_values,
                    max_truncation_error=max_truncation_error,
                    relative=relative)
  tensors = [Tensor(t, backend=backend) for t in out]
  return tuple(tensors)


def qr(
    tensor: Tensor,
    pivot_axis: int = -1,
    non_negative_diagonal: bool = False
) -> Tuple[Tensor, Tensor]:
  """
  QR reshapes tensor into a matrix and then decomposes that matrix into the
  product of unitary and upper triangular matrices Q and R. Q is reshaped
  into a tensor depending on the input shape and the choice of pivot_axis.

  Computes the reduced QR decomposition of the matrix formed by concatenating
  tensor about pivot_axis, e.g.
      ``` shape = tensor.shape
          columns = np.prod(shape[:pivot_axis])
          rows = np.prod(shape[pivot_axis:])
          matrix = tensor.reshape((columns, rows))
      ```
  The output is then shaped as follows:
     - Q has dimensions (*shape[:pivot_axis], np.prod(shape[pivot_axis:])).
     - R is a square matrix with length np.prod(shape[pivot_axis:]).

  The argument non_negative_diagonal, True by default, enforces a phase
  convention such that R has strictly non-negative entries on its main diagonal.
  This makes the QR decomposition unambiguous and unique, which allows
  it to be used in fixed point iterations. If False, the phase convention is set
  by the backend and thus undefined at the TN interface level, but this
  routine will be slightly less expensive.

  By default this pivot_axis is 1, which produces the usual behaviour in the
  matrix case.
  Args:
    tensor: The Tensor to be decomposed.
    pivot_axis: The axis of Tensor about which to concatenate.
                Default: 1
    non_negative_diagonal:


  Returns:
    Q, R : The decomposed Tensor with dimensions as specified above.
  """
  backend = tensor.backend
  out = backend.qr(tensor.array, pivot_axis=pivot_axis,
                   non_negative_diagonal=non_negative_diagonal)
  Q, R = [Tensor(t, backend=backend) for t in out]
  return Q, R


def rq(
    tensor: Tensor,
    pivot_axis: int = -1,
    non_negative_diagonal: bool = False
) -> Tuple[Tensor, Tensor]:
  """
  RQ reshapes tensor into a matrix and then decomposes that matrix into the
  product of upper triangular and unitary matrices R and Q. Q is reshaped
  into a tensor depending on the input shape and the choice of pivot_axis.

  Computes the reduced RQ decomposition of the matrix formed by concatenating
  tensor about pivot_axis, e.g.
      ``` shape = tensor.shape
          columns = np.prod(shape[:pivot_axis])
          rows = np.prod(shape[pivot_axis:])
          matrix = tensor.reshape((columns, rows))
      ```
  The output is then shaped as follows:
     - R is a square matrix with length np.prod(shape[:pivot_axis]).
     - Q has dimensions (np.prod(shape[:pivot_axis]), *shape[pivot_axis:]).

  The argument non_negative_diagonal, True by default, enforces a phase
  convention such that R has strictly non-negative entries on its main diagonal.
  This makes the RQ decomposition unambiguous and unique, which allows
  it to be used in fixed point iterations. If False, the phase convention is set
  by the backend and thus undefined at the TN interface level, but this
  routine will be slightly less expensive.

  By default this pivot_axis is 1, which produces the usual behaviour in the
  matrix case.
  Args:
    tensor: The Tensor to be decomposed.
    pivot_axis: The axis of Tensor about which to concatenate.
                Default: 1
    non_negative_diagonal:


  Returns:
    R, Q : The decomposed Tensor with dimensions as specified above.
  """
  backend = tensor.backend
  out = backend.rq(tensor.array, pivot_axis=pivot_axis,
                   non_negative_diagonal=non_negative_diagonal)
  R, Q = [Tensor(t, backend=backend) for t in out]
  return R, Q


def eigh(matrix: Tensor) -> Tuple[Tensor, Tensor]:
  """Compute eigenvectors and eigenvalues of a hermitian matrix.

  Args:
    matrix: A symetric matrix.
  Returns:
    Tensor: The eigenvalues in ascending order.
    Tensor: The eigenvectors.
  """
  backend = matrix.backend
  out = backend.eigh(matrix.array)
  tensors = [Tensor(t, backend=backend) for t in out]
  return tuple(tensors)


def norm(tensor: Tensor) -> Tensor:
  """Calculate the L2-norm of the elements of `tensor`
  """
  backend = tensor.backend
  out = backend.norm(tensor.array)
  return out


def inv(matrix: Tensor) -> Tensor:
  """Compute the matrix inverse of `matrix`.

  Args:
    matrix: A matrix.
  Returns:
    Tensor: The inverse of `matrix`
  """
  backend = matrix.backend
  out = backend.inv(matrix.array)
  tensor = Tensor(out, backend=backend)
  return tensor


def expm(matrix: Tensor) -> Tensor:
  """
  Return expm log of `matrix`, matrix exponential.
  Args:
    matrix: A tensor.
  Returns:
    Tensor
  """
  backend = matrix.backend
  out = backend.expm(matrix.array)
  tensor = Tensor(out, backend=backend)
  return tensor
