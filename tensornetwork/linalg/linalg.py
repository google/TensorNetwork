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

from typing import Any, Union, Text, Optional, List, Sequence, Tuple
from tensornetwork.tensor import Tensor

def svd(
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
  backend = tensor.backend
  out = backend.svd(tensor.array, split_axis,
                    max_singular_values=max_singular_values,
                    max_truncation_error=max_truncation_error,
                    relative=relative)
  tensors = [Tensor(t, backend=backend) for t in out]
  return tensors


def qr(
    tensor: Tensor,
    split_axis: int,
) -> Tuple[Tensor, Tensor]:
  """Computes the QR decomposition of a tensor."""
  backend = tensor.backend
  out = backend.qr(tensor.array, split_axis)
  tensors = [Tensor(t, backend=backend) for t in out]
  return tensors


def rq(
    tensor: Tensor,
    split_axis: int,
) -> Tuple[Tensor, Tensor]:
  """Computes the RQ (reversed QR) decomposition of a tensor."""
  backend = tensor.backend
  out = backend.rq(tensor.array, split_axis)
  tensors = [Tensor(t, backend=backend) for t in out]
  return tensors


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
  return tensors


def norm(tensor: Tensor) -> Tensor:
  """Calculate the L2-norm of the elements of `tensor`
  """
  backend = tensor.backend
  out = backend.norm(tensor.array)
  return out


def trace(tensor: Tensor) -> Tensor:
  """Calculate the trace over the last two axes of the given tensor."""
  raise NotImplementedError()


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
