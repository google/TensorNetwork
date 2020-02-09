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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.network_components import Node, contract, contract_between
# pylint: disable=line-too-long
from tensornetwork.backends import backend_factory
import copy
import warnings
from typing import List, Union, Any, Optional, Tuple, Text, Iterable, Type


class BaseCharge:

  def __init__(self,
               charges: np.ndarray,
               charge_labels: Optional[np.ndarray] = None,
               charge_types: Optional[List[Type["BaseCharge"]]] = None) -> None:
    self.charge_types = charge_types
    if charges.ndim == 1:
      charges = np.expand_dims(charges, 0)
    if charge_labels is None:
      self.unique_charges, self.charge_labels = np.unique(
          charges.astype(np.int16), return_inverse=True, axis=1)
      self.charge_labels = self.charge_labels.astype(np.int16)
    else:
      self.charge_labels = np.asarray(charge_labels, dtype=np.int16)
      # if charge_labels.dtype not in (np.int16, np.int16):
      #   raise TypeError("`charge_labels` have to be of dtype `np.int16`")

      self.unique_charges = charges.astype(np.int16)
      self.charge_labels = charge_labels.astype(np.int16)

  @staticmethod
  def fuse(charge1, charge2):
    raise NotImplementedError("`fuse` has to be implemented in derived classes")

  @staticmethod
  def dual_charges(charges):
    raise NotImplementedError(
        "`dual_charges` has to be implemented in derived classes")

  @staticmethod
  def identity_charge():
    raise NotImplementedError(
        "`identity_charge` has to be implemented in derived classes")

  @classmethod
  def random(cls, minval: int, maxval: int, dimension: int):
    raise NotImplementedError(
        "`random` has to be implemented in derived classes")

  @property
  def dim(self):
    return len(self.charge_labels)

  @property
  def num_symmetries(self) -> int:
    """
    Return the number of different charges in `ChargeCollection`.
    """
    return self.unique_charges.shape[0]

  @property
  def num_unique(self) -> int:
    """
    Return the number of different charges in `ChargeCollection`.
    """
    return self.unique_charges.shape[1]

  @property
  def identity_charges(self) -> np.ndarray:
    """
    Give the identity charge associated to a symmetries type in `charge_types`.
    Args:
      charge_types: A list of charge-types.
    Returns:
      nd.array: vector of identity charges for each symmetry in self 
    """
    unique_charges = np.expand_dims(
        np.asarray([ct.identity_charge() for ct in self.charge_types],
                   dtype=np.int16), 1)
    charge_labels = np.zeros(1, dtype=np.int16)
    obj = self.__new__(type(self))
    obj.__init__(unique_charges, charge_labels, self.charge_types)
    return obj

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `BaseCharge` object.
    Returns:
      Charge: The result of fusing `self` with `other`.
    """

    # fuse the unique charges from each index, then compute new unique charges
    comb_charges = fuse_ndarray_charges(self.unique_charges,
                                        other.unique_charges, self.charge_types)
    [unique_charges, charge_labels] = np.unique(
        comb_charges, return_inverse=True, axis=1)
    charge_labels = charge_labels.reshape(self.unique_charges.shape[1],
                                          other.unique_charges.shape[1]).astype(
                                              np.int16)

    # find new labels using broadcasting
    charge_labels = charge_labels[(
        self.charge_labels[:, None] + np.zeros([1, len(other)], dtype=np.int16)
    ).ravel(), (other.charge_labels[None, :] +
                np.zeros([len(self), 1], dtype=np.int16)).ravel()]

    obj = self.__new__(type(self))
    obj.__init__(unique_charges, charge_labels, self.charge_types)

    return obj

  def dual(self, take_dual: Optional[bool] = False) -> np.ndarray:
    if take_dual:
      unique_dual_charges = np.stack([
          self.charge_types[n].dual_charges(self.unique_charges[n, :])
          for n in range(len(self.charge_types))
      ],
                                     axis=0)

      obj = self.__new__(type(self))
      obj.__init__(unique_dual_charges, self.charge_labels, self.charge_types)
      return obj
    return self

  def copy(self):
    obj = self.__new__(type(self))
    obj.__init__(
        charges=self.unique_charges.copy(),
        charge_labels=self.charge_labels.copy(),
        charge_types=self.charge_types)
    return obj

  @property
  def charges(self):
    return self.unique_charges[:, self.charge_labels]

  def __repr__(self):
    return str(
        type(self)) + '\n' + 'charges: \n' + self.charges.__repr__() + '\n'

  def __len__(self):
    return len(self.charge_labels)

  def __mul__(self, number: bool) -> "BaseCharge":
    if not isinstance(number, (bool, np.bool_)):
      raise ValueError(
          "can only multiply by `True` or `False`, found {}".format(number))
    return self.dual(number)

  def intersect(self, other, assume_unique=False, return_indices=False) -> Any:
    if isinstance(other, type(self)):
      out = intersect(
          self.unique_charges,
          other.unique_charges,
          assume_unique=True,
          axis=1,
          return_indices=return_indices)
      obj = self.__new__(type(self))
    else:
      out = intersect(
          self.unique_charges,
          np.asarray(other),
          axis=1,
          assume_unique=assume_unique,
          return_indices=return_indices)
      obj = self.__new__(type(self))
    if return_indices == True:
      obj.__init__(
          charges=out[0],
          charge_labels=np.arange(len(out[0]), dtype=np.int16),
          charge_types=self.charge_types,
      )
      return obj, out[1], out[2]
    return obj

  def unique(self,
             return_index=False,
             return_inverse=False,
             return_counts=False):
    """
    Compute the unique charges in `BaseCharge`.
    See np.unique for a more detailed explanation. This function
    does the same but instead of a np.ndarray, it returns the unique
    elements in a `BaseCharge` object.
    Args:
      return_index: If `True`, also return the indices of `self.charges` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
      return_inverse: If `True`, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `self.charges`.
      return_counts: If `True`, also return the number of times each unique item appears
        in `self.charges`.
    Returns:
      BaseCharge: The sorted unique values.
      np.ndarray: The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
      np.ndarray: The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
      np.ndarray: The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.      
    """
    obj = self.__new__(type(self))
    tmp = np.unique(
        self.charge_labels,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts)
    if return_index or return_inverse or return_counts:
      if tmp[0].ndim == 0:
        index = np.asarray([tmp[0]])
        unique_charges = self.unique_charges[:, index]
      else:
        unique_charges = self.unique_charges[:, tmp[0]]
    else:
      if tmp.ndim == 0:
        tmp = np.asarray([tmp])
      unique_charges = self.unique_charges[:, tmp]
    obj.__init__(
        charges=unique_charges,
        charge_labels=np.arange(unique_charges.shape[1], dtype=np.int16),
        charge_types=self.charge_types)
    out = [obj]
    if return_index or return_inverse or return_counts:
      for n in range(1, len(tmp)):
        out.append(tmp[n])

    if len(out) == 1:
      return out[0]
    if len(out) == 2:
      return out[0], out[1]
    if len(out) == 3:
      return out[0], out[1], out[2]
    if len(out) == 4:
      return out[0], out[1], out[2], out[3]

  @property
  def dtype(self):
    return self.unique_charges.dtype

  @property
  def degeneracies(self):
    return np.sum(
        np.expand_dims(self.charge_labels, 1) == np.expand_dims(
            np.arange(self.unique_charges.shape[1], dtype=np.int16), 0),
        axis=0)

  def reduce(self,
             target_charges: np.ndarray,
             return_locations: bool = False,
             strides: int = 1) -> Any:
    """
    Reduce the dim of a charge to keep only the index values that intersect target_charges
    Args:
      target_charges: array of unique quantum numbers to keep.
      return_locations: If `True`, also return the output index 
        locations of target values.
    Returns:
      BaseCharge: index of reduced dimension.
      np.ndarray: If `return_locations = True`; the index locations 
        of target values.
    """
    if isinstance(target_charges, (np.integer, int)):
      target_charges = np.asarray([target_charges], dtype=np.int16)
    if target_charges.ndim == 1:
      target_charges = np.expand_dims(target_charges, 0)
    target_charges = np.asarray(target_charges, dtype=np.int16)
    # find intersection of index charges and target charges
    reduced_charges, label_to_unique, label_to_target = intersect(
        self.unique_charges, target_charges, axis=1, return_indices=True)
    num_unique = len(label_to_unique)

    # construct the map to the reduced charges
    map_to_reduced = np.full(self.dim, fill_value=-1, dtype=np.int16)
    map_to_reduced[label_to_unique] = np.arange(num_unique, dtype=np.int16)

    # construct the map to the reduced charges
    reduced_ind_labels = map_to_reduced[self.charge_labels]
    reduced_locs = reduced_ind_labels >= 0
    new_ind_labels = reduced_ind_labels[reduced_locs].astype(np.int16)
    obj = self.__new__(type(self))
    obj.__init__(reduced_charges, new_ind_labels, self.charge_types)

    if return_locations:
      return obj, strides * np.flatnonzero(reduced_locs).astype(np.uint32)
    return obj

  def __matmul__(self, other):
    #some checks
    if len(self) != len(other):
      raise ValueError(
          '__matmul__ requires charges to have the same number of elements')
    charges = np.concatenate([self.charges, other.charges], axis=0)
    charge_types = self.charge_types + other.charge_types
    return BaseCharge(
        charges=charges, charge_labels=None, charge_types=charge_types)

  def __getitem__(self, n: Union[np.ndarray, int]) -> "BaseCharge":
    """
    Return the charge-element at position `n`, wrapped into a `BaseCharge`
    object.
    Args:
      n: An integer or `np.ndarray`.
    Returns:
      BaseCharge: The charges at `n`.
    """

    if isinstance(n, (np.integer, int)):
      n = np.asarray([n])
    obj = self.__new__(type(self))
    labels = self.charge_labels[n]
    unique_labels, new_labels = np.unique(labels, return_inverse=True)
    if unique_labels.ndim == 0:
      unique_labels = np.asarray(unique_labels)
    unique_charges = self.unique_charges[:, unique_labels]
    obj.__init__(unique_charges, new_labels, self.charge_types)
    return obj

  def __eq__(self,
             target_charges: Union[np.ndarray, "BaseCharge"]) -> np.ndarray:

    if isinstance(target_charges, type(self)):
      targets = np.unique(
          target_charges.unique_charges[:, target_charges.charge_labels],
          axis=1)
    else:
      if target_charges.ndim == 1:
        target_charges = np.expand_dims(target_charges, 0)
      targets = np.unique(target_charges, axis=1)
    inds = np.nonzero(
        np.logical_and.reduce(
            np.expand_dims(self.unique_charges, 2) == np.expand_dims(
                targets, 1),
            axis=0))[0]
    return np.expand_dims(self.charge_labels, 1) == np.expand_dims(inds, 0)

  def isin(self, target_charges: Union[np.ndarray, "BaseCharge"]) -> np.ndarray:

    if isinstance(target_charges, type(self)):
      targets = target_charges.unique_charges
    else:
      targets = np.unique(target_charges, axis=1)
    tmp = np.expand_dims(self.unique_charges, 2) == np.expand_dims(targets, 1)
    inds = np.nonzero(
        np.logical_or.reduce(np.logical_and.reduce(tmp, axis=0), axis=1))[0]

    return np.isin(self.charge_labels, inds)


class U1Charge(BaseCharge):

  def __init__(self,
               charges: np.ndarray,
               charge_labels: Optional[np.ndarray] = None,
               charge_types: Optional[List[Type["BaseCharge"]]] = None) -> None:
    super().__init__(charges, charge_labels, charge_types=[type(self)])

  @staticmethod
  def fuse(charge1, charge2) -> np.ndarray:
    return np.add.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges) -> np.ndarray:
    return charges * charges.dtype.type(-1)

  @staticmethod
  def identity_charge() -> np.ndarray:
    return np.int16(0)

  @classmethod
  def random(cls, minval: int, maxval: int, dimension: int) -> np.ndarray:
    charges = np.random.randint(minval, maxval, dimension, dtype=np.int16)
    return cls(charges=charges)


def fuse_ndarray_charges(charges_A: np.ndarray, charges_B: np.ndarray,
                         charge_types: List[Type[BaseCharge]]) -> np.ndarray:
  """
  Fuse the quantum numbers of two indices under their kronecker addition.
  Args:
    charges_A (np.ndarray): n-by-D1 dimensional array integers encoding charges,
      with n the number of symmetries and D1 the index dimension.
    charges__B (np.ndarray): n-by-D2 dimensional array of charges.
    charge_types: A list of types of the charges.
  Returns:
    np.ndarray: n-by-(D1 * D2) dimensional array of the fused charges.
  """
  comb_charges = [0] * len(charge_types)
  for n in range(len(charge_types)):
    comb_charges[n] = charge_types[n].fuse(charges_A[n, :], charges_B[n, :])

  return np.concatenate(
      comb_charges, axis=0).reshape(
          len(charge_types), charges_A.shape[1] * charges_B.shape[1])


def intersect(A: np.ndarray,
              B: np.ndarray,
              axis=0,
              assume_unique=False,
              return_indices=False) -> Any:
  """
  Extends numpy's intersect1d to find the row or column-wise intersection of
  two 2d arrays. Takes identical input to numpy intersect1d.
  Args:
    A, B (np.ndarray): arrays of matching widths and datatypes
  Returns:
    ndarray: sorted 1D array of common rows/cols between the input arrays
    ndarray: the indices of the first occurrences of the common values in A.
      Only provided if return_indices is True.
    ndarray: the indices of the first occurrences of the common values in B.
      Only provided if return_indices is True.
  """
  #see https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
  if A.ndim == 1:
    return np.intersect1d(
        A, B, assume_unique=assume_unique, return_indices=return_indices)

  elif A.ndim == 2:
    if axis == 0:
      ncols = A.shape[1]
      if A.shape[1] != B.shape[1]:
        raise ValueError("array widths must match to intersect")

      dtype = {
          'names': ['f{}'.format(i) for i in range(ncols)],
          'formats': ncols * [A.dtype]
      }
      if return_indices:
        C, A_locs, B_locs = np.intersect1d(
            A.view(dtype),
            B.view(dtype),
            assume_unique=assume_unique,
            return_indices=return_indices)
        return C.view(A.dtype).reshape(-1, ncols), A_locs, B_locs
      C = np.intersect1d(
          A.view(dtype), B.view(dtype), assume_unique=assume_unique)
      return C.view(A.dtype).reshape(-1, ncols)

    elif axis == 1:
      #@Glen: why the copy here?
      out = intersect(
          A.T.copy(),
          B.T.copy(),
          axis=0,
          assume_unique=assume_unique,
          return_indices=return_indices)
      if return_indices:
        return out[0].T, out[1], out[2]
      return out.T

    else:
      raise NotImplementedError(
          "intersection can only be performed on first or second axis")

  else:
    raise NotImplementedError(
        "intersect is only implemented for 1d or 2d arrays")


def fuse_charges(charges: List[BaseCharge], flows: List[bool]) -> BaseCharge:
  """
  Fuse all `charges` into a new charge.
  Charges are fused from "right to left",
  in accordance with row-major order.

  Args:
    charges: A list of charges to be fused.
    flows: A list of flows, one for each element in `charges`.
  Returns:
    BaseCharge: The result of fusing `charges`.
  """
  if len(charges) != len(flows):
    raise ValueError(
        "`charges` and `flows` are of unequal lengths {} != {}".format(
            len(charges), len(flows)))
  fused_charges = charges[0] * flows[0]
  for n in range(1, len(charges)):
    fused_charges = fused_charges + charges[n] * flows[n]
  return fused_charges


def fuse_degeneracies(degen1: Union[List, np.ndarray],
                      degen2: Union[List, np.ndarray]) -> np.ndarray:
  """
  Fuse degeneracies `degen1` and `degen2` of two leg-charges
  by simple kronecker product. `degen1` and `degen2` typically belong to two
  consecutive legs of `BlockSparseTensor`.
  Given `degen1 = [1, 2, 3]` and `degen2 = [10, 100]`, this returns
  `[10, 100, 20, 200, 30, 300]`.
  When using row-major ordering of indices in `BlockSparseTensor`,
  the position of `degen1` should be "to the left" of the position of `degen2`.
  Args:
    degen1: Iterable of integers
    degen2: Iterable of integers
  Returns:
    np.ndarray: The result of fusing `dege1` with `degen2`.
  """
  return np.reshape(degen1[:, None] * degen2[None, :],
                    len(degen1) * len(degen2))


# class BaseCharge:

#   def __init__(self,
#                charges: np.ndarray,
#                charge_labels: Optional[np.ndarray] = None) -> None:
#     if charges.dtype != np.int16:
#       raise TypeError("`charges` have to be of dtype `np.int16`")

#     if charge_labels is None:

#       self.unique_charges, charge_labels = np.unique(
#           charges, return_inverse=True)
#       self.charge_labels = charge_labels.astype(np.int16)

#     else:
#       if charge_labels.dtype not in (np.int16, np.int16):
#         raise TypeError("`charge_labels` have to be of dtype `np.int16`")

#       self.unique_charges = charges
#       self.charge_labels = charge_labels.astype(np.int16)

#   def __add__(self, other: "BaseCharge") -> "BaseCharge":
#     # fuse the unique charges from each index, then compute new unique charges
#     comb_qnums = self.fuse(self.unique_charges, other.unique_charges)
#     [unique_charges, charge_labels] = np.unique(comb_qnums, return_inverse=True)
#     charge_labels = charge_labels.reshape(
#         len(self.unique_charges), len(other.unique_charges)).astype(np.int16)

#     # find new labels using broadcasting (could use np.tile but less efficient)
#     charge_labels = charge_labels[(
#         self.charge_labels[:, None] + np.zeros([1, len(other)], dtype=np.int16)
#     ).ravel(), (other.charge_labels[None, :] +
#                 np.zeros([len(self), 1], dtype=np.int16)).ravel()]
#     obj = self.__new__(type(self))
#     obj.__init__(unique_charges, charge_labels)
#     return obj

#   def __len__(self):
#     return len(self.charge_labels)

#   def dual(self, take_dual: Optional[bool] = False) -> "BaseCharge":
#     if take_dual:
#       obj = self.__new__(type(self))
#       obj.__init__(self.dual_charges(self.unique_charges), self.charge_labels)
#       return obj
#     return self

#   @property
#   def num_symmetries(self) -> int:
#     """
#     Return the number of different charges in `ChargeCollection`.
#     """
#     return self.unique_charges.shape[0]

#   def charges(self) -> np.ndarray:
#     return self.unique_charges[self.charge_labels]

#   @property
#   def dim(self):
#     return len(self.charge_labels)

#   @property
#   def dtype(self):
#     return self.unique_charges.dtype

#   def __repr__(self):
#     return str(
#         type(self)) + '\n' + 'charges: ' + self.charges().__repr__() + '\n'

#   @property
#   def degeneracies(self):
#     return np.sum(
#         np.expand_dims(self.charge_labels, 1) == np.expand_dims(
#             np.arange(len(self.unique_charges), dtype=np.int16), 0),
#         axis=0)

#   def unique(self,
#              return_index=False,
#              return_inverse=False,
#              return_counts=False
#             ) -> Tuple["BaseCharge", np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute the unique charges in `BaseCharge`.
#     See np.unique for a more detailed explanation. This function
#     does the same but instead of a np.ndarray, it returns the unique
#     elements in a `BaseCharge` object.
#     Args:
#       return_index: If `True`, also return the indices of `self.charges` (along the specified axis,
#         if provided, or in the flattened array) that result in the unique array.
#       return_inverse: If `True`, also return the indices of the unique array (for the specified
#         axis, if provided) that can be used to reconstruct `self.charges`.
#       return_counts: If `True`, also return the number of times each unique item appears
#         in `self.charges`.
#     Returns:
#       BaseCharge: The sorted unique values.
#       np.ndarray: The indices of the first occurrences of the unique values in the
#         original array. Only provided if `return_index` is True.
#       np.ndarray: The indices to reconstruct the original array from the
#         unique array. Only provided if `return_inverse` is True.
#       np.ndarray: The number of times each of the unique values comes up in the
#         original array. Only provided if `return_counts` is True.
#     """
#     obj = self.__new__(type(self))
#     obj.__init__(
#         self.unique_charges,
#         charge_labels=np.arange(len(self.unique_charges), dtype=np.int16))

#     out = [obj]
#     if return_index:
#       _, index = np.unique(self.charge_labels, return_index=True)
#       out.append(index)
#     if return_inverse:
#       out.append(self.charge_labels)
#     if return_counts:
#       out.append(self.degeneracies)
#     if len(out) == 1:
#       return out[0]
#     if len(out) == 2:
#       return out[0], out[1]
#     if len(out) == 3:
#       return out[0], out[1], out[2]

#   def isin(self, targets: Union[int, Iterable, "BaseCharge"]) -> np.ndarray:
#     """
#     Test each element of `BaseCharge` if it is in `targets`. Returns
#     an `np.ndarray` of `dtype=bool`.
#     Args:
#       targets: The test elements
#     Returns:
#       np.ndarray: An array of `bool` type holding the result of the comparison.
#     """
#     if isinstance(targets, type(self)):
#       targets = targets.unique_charges
#     targets = np.asarray(targets)
#     common, label_to_unique, label_to_targets = np.intersect1d(
#         self.unique_charges, targets, return_indices=True)
#     if len(common) == 0:
#       return np.full(len(self.charge_labels), fill_value=False, dtype=np.bool)
#     return np.isin(self.charge_labels, label_to_unique)

#   def __iter__(self):
#     return iter(self.charges())

#   def __getitem__(self, n: Union[np.ndarray, int]) -> "BaseCharge":
#     """
#     Return the charge-element at position `n`, wrapped into a `BaseCharge`
#     object.
#     Args:
#       n: An integer or `np.ndarray`.
#     Returns:
#       BaseCharge: The charges at `n`.
#     """

#     if isinstance(n, (np.integer, int)):
#       n = np.asarray([n])
#     obj = self.__new__(type(self))
#     obj.__init__(self.unique_charges, self.charge_labels[n])
#     return obj

#   def __mul__(self, number: bool) -> "BaseCharge":
#     if not isinstance(number, bool):
#       raise ValueError(
#           "can only multiply by `True` or `False`, found {}".format(number))

#     return self.dual(number)

#   def intersect(self, other, return_indices=False):
#     out = np.intersect1d(
#         self.unique_charges,
#         other.unique_charges,
#         assume_unique=True,
#         return_indices=return_indices)
#     obj = self.__new__(type(self))
#     if not return_indices:
#       obj.__init__(out, np.arange(len(out), dtype=np.int16))
#       return obj
#     obj.__init__(out[0], np.arange(len(out), dtype=np.int16))
#     return obj, out[1], out[2]

#   def reduce(self,
#              target_charges: np.ndarray,
#              return_locs: bool = False,
#              strides: int = 1) -> ("SymIndex", np.ndarray):
#     """
#     Reduce the dim of a SymIndex to keep only the index values that intersect target_charges
#     Args:
#       target_charges (np.ndarray): array of unique quantum numbers to keep.
#       return_locs (bool, optional): if True, also return the output index
#         locations of target values.
#     Returns:
#       SymIndex: index of reduced dimension.
#       np.ndarray: output index locations of target values.
#     """
#     if isinstance(target_charges, (int, np.integer)):
#       target_charges = np.asarray([target_charges])
#     target_charges = np.asarray(target_charges, dtype=np.int16)
#     # find intersection of index charges and target charges
#     reduced_charges, label_to_unique, label_to_target = intersect(
#         self.unique_charges, target_charges, axis=1, return_indices=True)

#     num_unique = len(label_to_unique)

#     # construct the map to the reduced charges
#     map_to_reduced = np.full(self.dim, fill_value=-1, dtype=np.int16)
#     map_to_reduced[label_to_unique] = np.arange(num_unique, dtype=np.int16)

#     # construct the map to the reduced charges
#     reduced_ind_labels = map_to_reduced[self.charge_labels]
#     reduced_locs = reduced_ind_labels >= 0
#     new_ind_labels = reduced_ind_labels[reduced_locs].astype(np.int16)
#     obj = self.__new__(type(self))

#     obj.__init__(reduced_charges, new_ind_labels)
#     if return_locs:
#       return obj, strides * np.flatnonzero(reduced_locs).astype(np.uint32)
#     return obj

# class ChargeCollection:

#   def __init__(self,
#                charge_types: List[Type[BaseCharge]],
#                charges: np.ndarray,
#                charge_labels: Optional[np.ndarray] = None) -> None:
#     self.charge_types = charge_types
#     if charges.ndim == 1:
#       charges = np.expand_dims(charges, 0)
#     if charge_labels is None:
#       self.unique_charges, self.charge_labels = np.unique(
#           charges.astype(np.int16), return_inverse=True, axis=1)
#       self.charge_labels = self.charge_labels.astype(np.int16)
#     else:
#       if charge_labels.dtype not in (np.int16, np.int16):
#         raise TypeError("`charge_labels` have to be of dtype `np.int16`")

#       self.unique_charges = charges.astype(np.int16)
#       self.charge_labels = charge_labels.astype(np.int16)

#   @property
#   def dim(self):
#     return len(self.charge_labels)

#   @property
#   def num_symmetries(self) -> int:
#     """
#     Return the number of different charges in `ChargeCollection`.
#     """
#     return self.unique_charges.shape[0]

#   @property
#   def identity_charges(self) -> np.ndarray:
#     """
#     Give the identity charge associated to a symmetries type in `charge_types`.
#     Args:
#       charge_types: A list of charge-types.
#     Returns:
#       nd.array: vector of identity charges for each symmetry in self
#     """
#     unique_charges = np.expand_dims(
#         np.asarray([ct.identity for ct in self.charge_types], dtype=np.int16),
#         1)
#     charge_labels = np.zeros(1, dtype=np.int16)
#     obj = self.__new__(type(self))
#     obj.__init__(unique_charges, charge_labels)
#     return obj

#   def __add__(self, other: "ChargeCollection") -> "ChargeCollection":
#     """
#     Fuse `self` with `other`.
#     Args:
#       other: A `ChargeCollection` object.
#     Returns:
#       Charge: The result of fusing `self` with `other`.
#     """

#     # fuse the unique charges from each index, then compute new unique charges
#     comb_charges = fuse_ndarray_charges(self.unique_charges,
#                                         other.unique_charges, self.charge_types)
#     [unique_charges, charge_labels] = np.unique(
#         comb_charges, return_inverse=True, axis=1)
#     charge_labels = charge_labels.reshape(self.unique_charges.shape[1],
#                                           other.unique_charges.shape[1]).astype(
#                                               np.int16)

#     # find new labels using broadcasting
#     charge_labels = charge_labels[(
#         self.charge_labels[:, None] + np.zeros([1, len(other)], dtype=np.int16)
#     ).ravel(), (other.charge_labels[None, :] +
#                 np.zeros([len(self), 1], dtype=np.int16)).ravel()]
#     return ChargeCollection(self.charge_types, unique_charges, charge_labels)

#   def dual(self, take_dual: Optional[bool] = False) -> np.ndarray:
#     if take_dual:
#       unique_dual_charges = np.stack([
#           self.charge_types[n].dual_charges(self.unique_charges[n, :])
#           for n in range(len(self.charge_types))
#       ],
#                                      axis=0)
#       return ChargeCollection(self.charge_types, unique_dual_charges,
#                               self.charge_labels)
#     return self

#   def charges(self):
#     return self.unique_charges[:, self.charge_labels]

#   def __repr__(self):
#     return str(
#         type(self)) + '\n' + 'charges: \n' + self.charges().__repr__() + '\n'

#   def __len__(self):
#     return len(self.charge_labels)

#   def __mul__(self, number: bool) -> "ChargeCollection":
#     if not isinstance(number, bool):
#       raise ValueError(
#           "can only multiply by `True` or `False`, found {}".format(number))
#     return self.dual(number)

#   def unique(self,
#              return_index=False,
#              return_inverse=False,
#              return_counts=False
#             ) -> Tuple["BaseCharge", np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute the unique charges in `BaseCharge`.
#     See np.unique for a more detailed explanation. This function
#     does the same but instead of a np.ndarray, it returns the unique
#     elements in a `BaseCharge` object.
#     Args:
#       return_index: If `True`, also return the indices of `self.charges` (along the specified axis,
#         if provided, or in the flattened array) that result in the unique array.
#       return_inverse: If `True`, also return the indices of the unique array (for the specified
#         axis, if provided) that can be used to reconstruct `self.charges`.
#       return_counts: If `True`, also return the number of times each unique item appears
#         in `self.charges`.
#     Returns:
#       BaseCharge: The sorted unique values.
#       np.ndarray: The indices of the first occurrences of the unique values in the
#         original array. Only provided if `return_index` is True.
#       np.ndarray: The indices to reconstruct the original array from the
#         unique array. Only provided if `return_inverse` is True.
#       np.ndarray: The number of times each of the unique values comes up in the
#         original array. Only provided if `return_counts` is True.
#     """

#     obj = ChargeCollection(
#         self.charge_types,
#         self.unique_charges,
#         charge_labels=np.arange(self.unique_charges.shape[1], dtype=np.int16))

#     out = [obj]
#     if return_index:
#       _, index = np.unique(self.charge_labels, return_index=True)
#       out.append(index)
#     if return_inverse:
#       out.append(self.charge_labels)
#     if return_counts:
#       _, cnts = np.unique(self.charge_labels, return_counts=True)
#       out.append(cnts)
#     if len(out) == 1:
#       return out[0]
#     if len(out) == 2:
#       return out[0], out[1]
#     if len(out) == 3:
#       return out[0], out[1], out[2]

#   @property
#   def dtype(self):
#     return self.unique_charges.dtype

#   @property
#   def degeneracies(self):
#     return np.sum(
#         np.expand_dims(self.charge_labels, 1) == np.expand_dims(
#             np.arange(self.unique_charges.shape[1], dtype=np.int16), 0),
#         axis=0)

#   def reduce(self,
#              target_charges: np.ndarray,
#              return_locs: bool = False,
#              strides: int = 1) -> ("SymIndex", np.ndarray):
#     """
#     Reduce the dim of a SymIndex to keep only the index values that intersect target_charges
#     Args:
#       target_charges (np.ndarray): array of unique quantum numbers to keep.
#       return_locs (bool, optional): if True, also return the output index
#         locations of target values.
#     Returns:
#       SymIndex: index of reduced dimension.
#       np.ndarray: output index locations of target values.
#     """
#     target_charges = np.asarray(target_charges, dtype=np.int16)
#     # find intersection of index charges and target charges
#     reduced_charges, label_to_unique, label_to_target = intersect(
#         self.unique_charges, target_charges, axis=1, return_indices=True)
#     num_unique = len(label_to_unique)

#     # construct the map to the reduced charges
#     map_to_reduced = np.full(self.dim, fill_value=-1, dtype=np.int16)
#     map_to_reduced[label_to_unique] = np.arange(num_unique, dtype=np.int16)

#     # construct the map to the reduced charges
#     reduced_ind_labels = map_to_reduced[self.charge_labels]
#     reduced_locs = reduced_ind_labels >= 0
#     new_ind_labels = reduced_ind_labels[reduced_locs].astype(np.int16)

#     if return_locs:
#       return ChargeCollection(
#           self.charge_types, reduced_charges,
#           new_ind_labels), strides * np.flatnonzero(reduced_locs).astype(
#               np.uint32)
#     return ChargeCollection(self.charge_types, reduced_charges, new_ind_labels)
