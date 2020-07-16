"""
This module contains functions that return some particular matrix or operator,
notably including Pauli matrices and Hamiltonians.
"""
from typing import Optional, Sequence, Tuple, Union, Callable, List, Text, Any
import tensornetwork as tn
from tensornetwork.backends import abstract_backend
from tensornetwork import backend_contextmanager, backends

import numpy as np

BackendType = Union[Text, abstract_backend.AbstractBackend]
DtypeType = Any
###############################################################################
# PAULI MATRICES
###############################################################################


def sigX(backend: Optional[BackendType] = None,
         dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  Pauli X matrix.

  Args:
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    out    : The Pauli X matrix, a tn.Tensor.
  """
  vals = [[0, 1],
          [1, 0]]
  array = np.array(vals, dtype=dtype)
  out = tn.Tensor(array, backend=backend)
  return out


def sigY(backend: Optional[BackendType] = None,
         dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  Pauli Y matrix.

  Args:
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    Y      : The Pauli Y matrix, a tn.Tensor.
  """
  vals = [[0, -1],
          [1, 0]]
  array = 1.0j*np.array(vals, dtype=dtype)
  out = tn.Tensor(array, backend=backend)
  return out


def sigZ(backend: Optional[BackendType] = None,
         dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  Pauli Z matrix.

  Args:
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    Z      : The Pauli Z matrix, a tn.Tensor.
  """
  vals = [[1, 0],
          [0, -1]]
  array = np.array(vals, dtype=dtype)
  out = tn.Tensor(array, backend=backend)
  return out


def sigU(backend: Optional[BackendType] = None,
         dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  Pauli 'up' matrix.

  Args:
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    U      : The Pauli U matrix, a tn.Tensor.
  """
  vals = [[0, 1],
          [0, 0]]
  array = np.array(vals, dtype=dtype)
  out = tn.Tensor(array, backend=backend)
  return out


def sigD(backend: Optional[BackendType] = None,
         dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  Pauli 'down' matrix.

  Args:
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    D      : The Pauli D matrix, a tn.Tensor.
  """
  vals = [[0, 0],
          [1, 0]]
  array = np.array(vals, dtype=dtype)
  out = tn.Tensor(array, backend=backend)
  return out


###############################################################################
# HAMILTONIANS
###############################################################################

def H_ising(h: float, J: float = 1., backend: Optional[BackendType] = None,
            dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  The famous and beloved transverse-field Ising model,
  H = J * XX + h * ZI

  Args:
    h  : Transverse field.
    J  : Coupling strength, default 1.
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    H      : The Hamiltonian, a tn.Tensor shape (2, 2, 2, 2).
  """
  X = sigX(backend=backend, dtype=dtype)
  Z = sigZ(backend=backend, dtype=dtype)
  Id = tn.eye(2, backend=backend, dtype=dtype)
  ham = J*tn.outer_product(X, X) + h*tn.outer_product(Z, Id)
  return ham


def H_XXZ(delta: float = 1, ud: float = 2., scale: float = 1.,
          backend: Optional[BackendType] = None,
          dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  H = (-1/(8scale))*[ud*[UD + DU] + delta*ZZ]

  Args:
    delta, ud, scale: Couplings.
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    H      : The Hamiltonian, a tn.Tensor shape (2, 2, 2, 2).
  """
  U = sigU(backend=backend, dtype=dtype)
  D = sigD(backend=backend, dtype=dtype)
  Z = sigZ(backend=backend, dtype=dtype)
  UD = ud * (tn.outer_product(U, D) + tn.outer_product(D, U))
  H = UD + delta * tn.outer_product(Z, Z)
  H *= -(1/(8*scale))
  return H


def H_XX(backend: Optional[BackendType] = None,
         dtype: Optional[DtypeType] = None) -> tn.Tensor:
  """
  Args:
    backend: The backend.
    dtype  : dtype of data.
  Returns:
    H      : The Hamiltonian, a tn.Tensor shape (2, 2, 2, 2).
  """
  X = sigX(backend=backend, dtype=dtype)
  Y = sigY(backend=backend, dtype=dtype)
  H = tn.outer_product(X, X) + tn.outer_product(Y, Y)
  return H
