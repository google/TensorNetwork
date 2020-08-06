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

import numpy as np
from tensornetwork.block_sparse.utils import intersect
from typing import (List, Optional, Type, Any, Union, Callable)

#TODO (mganahl): switch from column to row order for unique labels
#TODO (mganahl): implement more efficient unique function
#TODO (mganahl): clean up implementation of identity charges
#TODO (mganahl): for rank-3 tensors with small bond dimensions, finding
#                blocks brute force is much faster. Implement this.


class BaseCharge:
  """
  Base class for charges of BlockSparseTensor. All user defined charges 
  should be derived from this class.
  Attributes:
    * unique_charges: np.ndarray of shape `(m,n)` with `m`
      the number of charge types, and `n` the number of unique charges.
    * charge_labels: np.ndarray of dtype np.int16. Used for identifying 
      charges with integer labels. `unique_charges[:, charge_labels] 
      is the np.ndarray of actual charges.
    * charge_types: A list of `type` objects. Stored the different charge types,
      on for each row in `unique_charges`.
      
  """

  class Iterator:
    def __init__(self, unique: np.ndarray, labels: np.ndarray):
      self.n = 0
      self.unique = unique
      self.labels = labels

    def __next__(self):
      if self.n < len(self.labels):
        out = self.unique[self.labels[self.n], :]
        self.n += 1
        return out
      raise StopIteration

  def __init__(self,
               charges: Union[List, np.ndarray],
               charge_labels: Optional[np.ndarray] = None,
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               charge_dtype: Optional[Type[np.number]] = np.int16) -> None:
    charges = np.asarray(charges)
    if charges.ndim == 1:
      charges = charges[:, None]

    if (charge_types is not None) and (len(charge_types) != charges.shape[1]):
      raise ValueError(
          "`len(charge_types) = {}` does not match `charges.shape[1]={}`"
          .format(len(charge_types), charges.shape[1]))

    if charges.shape[1] < 3:
      label_dtype = np.int16
    else:
      label_dtype = np.int32
    if charge_types is None:
      charge_types = [type(self)] * charges.shape[1]
    self.charge_types = charge_types
    if charge_labels is None:
      if charges.shape[0] > 0:
        self.unique_charges, self.charge_labels = np.unique(
            charges.astype(charge_dtype), return_inverse=True, axis=0)
        self.charge_labels = self.charge_labels.astype(label_dtype)
      else:
        self.unique_charges = np.empty((0, charges.shape[1]),
                                       dtype=charge_dtype)
        self.charge_labels = np.empty(0, dtype=label_dtype)
    else:
      self.charge_labels = np.asarray(charge_labels, dtype=label_dtype)
      self.unique_charges = charges.astype(charge_dtype)
      
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
  def random(cls, dimension: int, minval: int, maxval: int):
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
    return self.unique_charges.shape[1]

  @property
  def num_unique(self) -> int:
    """
    Return the number of different charges in `ChargeCollection`.
    """
    return self.unique_charges.shape[0]

  def copy(self):
    """
    Return a copy of `BaseCharge`.
    """
    obj = self.__new__(type(self))
    obj.__init__(
        charges=self.unique_charges.copy(),
        charge_labels=self.charge_labels.copy(),
        charge_types=self.charge_types,
        charge_dtype=self.dtype)
    return obj

  @property
  def charges(self):
    """
    Return the actual charges of `BaseCharge` as np.ndarray.
    """
    return self.unique_charges[self.charge_labels, :]

  @property
  def dtype(self):
    return self.unique_charges.dtype

  @property
  def label_dtype(self):
    return self.charge_labels.dtype

  def __repr__(self):
    return 'BaseCharge object:' + '\n   charge types: ' + self.names + \
        '\n   unique charges:' + str(self.charges.T).replace('\n', '\n\t ')\
        + '\n'

  def __iter__(self):
    return self.Iterator(self.unique_charges, self.charge_labels)

  def __len__(self):
    return len(self.charge_labels)

  def __eq__(self, target_charges: Union[np.ndarray,
                                         "BaseCharge"]) -> np.ndarray:
    #FIXME (mganahl): calling np.unique can cause significant overhead
    # in some cases. fix code in block_tensor.py to work on
    # np.ndarray instead
    if isinstance(target_charges, type(self)):
      if len(target_charges) == 0:
        raise ValueError('input to __eq__ cannot be an empty charge')
      targets = np.unique(
          target_charges.unique_charges[target_charges.charge_labels, :],
          axis=0)
    else:
      if target_charges.ndim == 1:
        target_charges = target_charges[:, None]
      if target_charges.shape[0] == 0:
        raise ValueError('input to __eq__ cannot be an empty np.ndarray')
      if target_charges.shape[1] != self.num_symmetries:
        raise ValueError(
            "shape of `target_charges = {}` is incompatible with "
            "`self.num_symmetries = {}"
            .format(target_charges.shape, self.num_symmetries))
      targets = np.unique(target_charges, axis=0)
    #pylint: disable=no-member
    inds = np.nonzero(
        np.logical_and.reduce(
            self.unique_charges[:, :, None] == targets.T[None, :, :],
            axis=1))[0]

    return self.charge_labels[:, None] == inds[None, :]

  def identity_charges(self, dim: int = 1) -> "BaseCharge":
    """
    Returns the identity charge.
    Returns:
      BaseCharge: The identity charge.
    """
    unique_charges = np.asarray(
        [ct.identity_charge() for ct in self.charge_types],
        dtype=self.dtype)[None, :]
    charge_labels = np.zeros(dim, dtype=self.label_dtype)
    obj = self.__new__(type(self))
    obj.__init__(unique_charges, charge_labels, self.charge_types)
    return obj

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `BaseCharge` object.
    Returns:
      BaseCharge: The result of fusing `self` with `other`.
    """
    if len(self.charge_labels) == 0 or len(other.charge_labels) == 0:
      obj = self.__new__(type(self))
      obj.__init__(
          np.empty((0, self.num_symmetries), dtype=self.dtype),
          np.empty(0, dtype=self.label_dtype), self.charge_types)
      return obj

    # fuse the unique charges from each index, then compute new unique charges
    comb_charges = fuse_ndarray_charges(self.unique_charges,
                                        other.unique_charges, self.charge_types)
    #pylint: disable=unsubscriptable-object
    if comb_charges.shape[0] == 0:
      obj = self.__new__(type(self))
      obj.__init__(
          np.empty((0, self.num_symmetries), dtype=self.dtype),
          np.empty(0, dtype=self.label_dtype), self.charge_types)
      return obj

    unique_charges, charge_labels = np.unique(
        comb_charges, return_inverse=True, axis=0)
    charge_labels = charge_labels.reshape(self.unique_charges.shape[0],
                                          other.unique_charges.shape[0]).astype(
                                              self.label_dtype)
    # find new labels using broadcasting
    left_labels = self.charge_labels[:, None] + np.zeros([1, len(other)],
                                                         dtype=self.label_dtype)
    right_labels = other.charge_labels[None, :] + np.zeros(
        [len(self), 1], dtype=self.label_dtype)
    charge_labels = charge_labels[np.ravel(left_labels), np.ravel(right_labels)]

    obj = self.__new__(type(self))
    obj.__init__(unique_charges, charge_labels, self.charge_types)

    return obj

  def dual(self, take_dual: Optional[bool] = False) -> "BaseCharge":
    """
    Return the charges of `BaseCharge`, possibly conjugated.
    Args:
      take_dual: If `True` return the dual charges. If `False` return 
        regular charges.
    Returns:
      BaseCharge
    """
    if take_dual:
      unique_dual_charges = np.stack([
          self.charge_types[n].dual_charges(self.unique_charges[:, n])
          for n in range(len(self.charge_types))
      ],
                                     axis=1)

      obj = self.__new__(type(self))
      obj.__init__(unique_dual_charges, self.charge_labels, self.charge_types)
      return obj
    return self

  def __matmul__(self, other):
    #some checks
    if len(self) != len(other):
      raise ValueError(
          '__matmul__ requires charges to have the same number of elements')
    charges = np.concatenate([self.charges, other.charges], axis=1)
    charge_types = self.charge_types + other.charge_types
    return BaseCharge(
        charges=charges, charge_labels=None, charge_types=charge_types)

  def __mul__(self, number: bool) -> "BaseCharge":
    if not isinstance(number, (bool, np.bool_)):
      raise ValueError(
          "can only multiply by `True` or `False`, found {}".format(number))
    return self.dual(number)

  def intersect(self, other, assume_unique=False, return_indices=False) -> Any:
    """
    Compute the intersection of `self` with `other`. See also np.intersect1d.
    Args:
      other: A BaseCharge object.
      assume_unique: If `True` assume that elements are unique.
      return_indices: If `True`, return index-labels.
    Returns:
      If `return_indices=True`:
        BaseCharge
        np.ndarray: The indices of the first occurrences of the 
          common values in `self`.
        np.ndarray: The indices of the first occurrences of the 
          common values in `other`.
      If `return_indices=False`:
        BaseCharge
    """
    if isinstance(other, type(self)):
      out = intersect(
          self.charges,
          other.charges,
          assume_unique=assume_unique,
          axis=0,
          return_indices=return_indices)
    else:
      if other.ndim == 1:
        other = other[:, None]
      out = intersect(
          self.charges,
          np.asarray(other),
          axis=0,
          assume_unique=assume_unique,
          return_indices=return_indices)
    obj = self.__new__(type(self))
    if return_indices:
      obj.__init__(
          charges=out[0],
          charge_labels=np.arange(out[0].shape[0], dtype=self.label_dtype),
          charge_types=self.charge_types,
      )
      return obj, out[1], out[2]
    obj.__init__(
        charges=out,
        charge_labels=np.arange(out.shape[0], dtype=self.label_dtype),
        charge_types=self.charge_types,
    )
    return obj

  def sort_unique_charges(self) -> "BaseCharge":
    """
    Sort the `unique_charges` of BaseCharge` according to standard order 
    used by numpy.
    Returns:
      BaseCharge
    """
    unique_charges, inverse = np.unique(
        self.unique_charges, return_inverse=True, axis=0)
    charge_labels = inverse[self.charge_labels]
    obj = self.__new__(type(self))
    obj.__init__(
        charges=unique_charges,
        charge_labels=charge_labels,
        charge_types=self.charge_types)
    return obj

  def unique(self,
             return_index: bool = False,
             return_inverse: bool = False,
             return_counts: bool = False,
             sort: bool = True) -> Any:
    """
    Compute the unique charges in `BaseCharge`.
    See np.unique for a more detailed explanation. This function
    does the same but instead of a np.ndarray, it returns the unique
    elements in a `BaseCharge` object.

    Args:
      return_index: If `True`, also return the indices of `self.charges` 
        (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
      return_inverse: If `True`, also return the indices of the unique array 
        (for the specified
        axis, if provided) that can be used to reconstruct `self.charges`.
      return_counts: If `True`, also return the number of times each unique 
        item appears in `self.charges`.
      sort: If `True`, the returned `BaseCharge` object has sorted 
        `unique_charges`. If `False`, `unique_`charges` are in general 
        not sorted.
    Returns:
      BaseCharge: The sorted unique values.
      np.ndarray: The indices of the first occurrences of the unique values 
        in the original array. Only provided if `return_index` is True.
      np.ndarray: The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
      np.ndarray: The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.      
    """

    if sort:
      #make sure that unique_charges are sorted
      tmp_charge = self.sort_unique_charges()
    else:
      tmp_charge = self
    obj = tmp_charge.__new__(type(tmp_charge))
    tmp = np.unique(
        tmp_charge.charge_labels,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts)
    if return_index or return_inverse or return_counts:
      if tmp[0].ndim == 0:  #only a single entry
        index = np.asarray([tmp[0]])
        unique_charges = tmp_charge.unique_charges[index, :]
      else:
        unique_charges = tmp_charge.unique_charges[tmp[0], :]
    else:
      if tmp.ndim == 0:
        tmp = np.asarray([tmp])
      unique_charges = tmp_charge.unique_charges[tmp, :]
    obj.__init__(
        charges=unique_charges,
        charge_labels=np.arange(
            unique_charges.shape[0], dtype=self.label_dtype),
        charge_types=self.charge_types)
    out = [obj]
    if return_index or return_inverse or return_counts:
      for n in range(1, len(tmp)):
        out.append(tmp[n])
    #for a single return value we don't want to return a list or tuple
    if len(out) == 1:
      return out[0]
    return tuple(out)

  def reduce(self,
             target_charges: np.ndarray,
             return_locations: bool = False,
             strides: Optional[int] = 1) -> Any:
    """
    Reduce the dimension of a 
    charge to keep only the charge values that intersect target_charges
    Args:
      target_charges: array of unique charges to keep.
      return_locations: If `True`, also return the locations of 
        target values within `BaseCharge`.
      strides: An optional stride value.
    Returns:
      BaseCharge: charge of reduced dimension.
      np.ndarray: If `return_locations = True`; the index locations 
        of target values.
    """
    if isinstance(target_charges, (np.integer, int)):
      target_charges = np.asarray([target_charges], dtype=self.dtype)
    if target_charges.ndim == 1:
      target_charges = target_charges[:, None]
    target_charges = np.asarray(target_charges, dtype=self.dtype)
    # find intersection of index charges and target charges
    reduced_charges, label_to_unique, _ = intersect(
        self.unique_charges, target_charges, axis=0, return_indices=True)
    num_unique = len(label_to_unique)

    # construct the map to the reduced charges
    map_to_reduced = np.full(self.dim, fill_value=-1, dtype=self.label_dtype)
    map_to_reduced[label_to_unique] = np.arange(
        num_unique, dtype=self.label_dtype)

    # construct the map to the reduced charges
    reduced_ind_labels = map_to_reduced[self.charge_labels]
    reduced_locs = reduced_ind_labels >= 0
    new_ind_labels = reduced_ind_labels[reduced_locs].astype(self.label_dtype)
    obj = self.__new__(type(self))
    obj.__init__(reduced_charges, new_ind_labels, self.charge_types)

    if return_locations:
      return obj, strides * np.flatnonzero(reduced_locs).astype(np.uint32)
    return obj

  def __getitem__(self, n: Union[List[int], np.ndarray, int]) -> "BaseCharge":
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
    n = np.asarray(n)
    obj = self.__new__(type(self))
    labels = self.charge_labels[n]
    unique_labels, new_labels = np.unique(labels, return_inverse=True)
    if unique_labels.ndim == 0:
      unique_labels = np.asarray(unique_labels)
    unique_charges = self.unique_charges[unique_labels, :]
    obj.__init__(unique_charges, new_labels, self.charge_types)
    return obj

  def isin(self, target_charges: Union[np.ndarray, "BaseCharge"]) -> np.ndarray:
    """
    See also np.isin. 
    Returns an np.ndarray of `dtype=bool`, with `True` at all linear positions
    where `self` is in `target_charges`, and `False` everywhere else.
    Args:
      target_charges: A `BaseCharge` object.
    Returns:
      np.ndarray: An array of boolean values.
    """
    if isinstance(target_charges, type(self)):
      if not np.all([
          a == b for a, b in zip(self.charge_types, target_charges.charge_types)
      ]):
        raise TypeError(
            "isin only callable for equal charge types, found {} and {}".format(
                self.charge_types, target_charges.charge_types))

      targets = target_charges.unique_charges
    else:
      if target_charges.ndim == 1:
        if target_charges.shape[0] == 0:
          raise ValueError("input to `isin` cannot be an empty np.ndarray")
        targets = np.unique(target_charges, axis=0)[:, None]
      elif target_charges.ndim == 2:
        if target_charges.shape[0] == 0:
          raise ValueError("input to `isin` cannot be an empty np.ndarray")

        targets = np.unique(target_charges, axis=0)
      else:
        raise ValueError("targets.ndim has to be 1 or 2, found {}".format(
            target_charges.ndim))
      if targets.shape[1] != self.num_symmetries:
        raise ValueError(
            "target_charges.shape[0]={} is different from "
            "self.num_symmetries = {}"
            .format(targets.shape[0], self.num_symmetries))

    tmp = self.unique_charges[:, :, None] == targets.T[None, :, :]
    #pylint: disable=no-member
    inds = np.nonzero(
        np.logical_or.reduce(np.logical_and.reduce(tmp, axis=1), axis=1))[0]

    return np.isin(self.charge_labels, inds)

  @property
  def names(self):
    return repr([ct.__new__(ct).__class__.__name__ for ct in self.charge_types])


class U1Charge(BaseCharge):
  """Charge Class for the U1 symmetry group."""

  @staticmethod
  def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
    return np.add.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges: np.ndarray) -> np.ndarray:
    return charges * charges.dtype.type(-1)

  @staticmethod
  def identity_charge() -> np.ndarray:
    return np.int16(0)

  @classmethod
  def random(cls, dimension: int, minval: int, maxval: int) -> BaseCharge:
    charges = np.random.randint(minval, maxval + 1, dimension, dtype=np.int16)
    return cls(charges=charges)


class Z2Charge(BaseCharge):
  """Charge Class for the Z2 symmetry group."""

  def __init__(self,
               charges: Union[List, np.ndarray],
               charge_labels: Optional[np.ndarray] = None,
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               charge_dtype: Optional[Type[np.number]] = np.int16) -> None:
    #do some checks before calling the base class constructor
    unique = np.unique(np.ravel(charges))
    if not np.all(np.isin(unique, [0, 1])):
      raise ValueError("Z2 charges can only be 0 or 1, found {}".format(unique))
    super().__init__(
        charges,
        charge_labels,
        charge_types=[type(self)],
        charge_dtype=charge_dtype)

  @staticmethod
  def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
    #pylint: disable=no-member
    return np.bitwise_xor.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges: np.ndarray) -> np.ndarray:
    return charges

  @staticmethod
  def identity_charge() -> np.ndarray:
    return np.int16(0)

  @classmethod
  def random(cls,
             dimension: int,
             minval: int = 0,
             maxval: int = 1) -> BaseCharge:
    if minval != 0 or maxval != 1:
      raise ValueError("Z2 charges can only take values 0 or 1")

    charges = np.random.randint(0, 2, dimension, dtype=np.int16)
    return cls(charges=charges)


def ZNCharge(n: int) -> Callable:
  """Contstructor for charge classes of the ZN symmetry groups.

  Args:
    n: The module of the symmetry group.
  Returns:
    A charge class of your given ZN symmetry group.
  """
  if n < 2:
    raise ValueError(f"n must be >= 2, found {n}")

  class ModularCharge(BaseCharge):

    def __init__(self,
                 charges: Union[List, np.ndarray],
                 charge_labels: Optional[np.ndarray] = None,
                 charge_types: Optional[List[Type["BaseCharge"]]] = None,
                 charge_dtype: Optional[Type[np.number]] = np.int16) -> None:
      unique = np.unique(np.ravel(charges))
      if not np.all(np.isin(unique, list(range(n)))):
        raise ValueError(f"Z{n} charges must be in range({n}), found: {unique}")
      super().__init__(
          charges,
          charge_labels,
          charge_types=[type(self)],
          charge_dtype=charge_dtype)

    @staticmethod
    def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
      #pylint: disable=no-member
      return np.outer(charge1, charge2).ravel() % n

    @staticmethod
    def dual_charges(charges: np.ndarray) -> np.ndarray:
      return (n - charges) % n

    @staticmethod
    def identity_charge() -> np.ndarray:
      return np.int16(0)

    @classmethod
    def random(cls,
               dimension: int,
               minval: int = 0,
               maxval: int = n - 1) -> BaseCharge:
      if maxval >= n:
        raise ValueError(f"maxval must be less than n={n}, got {maxval}")
      if minval < 0:
        raise ValueError(f"minval must be greater than 0, found {minval}")
      # No need for the mod due to the checks above.
      charges = np.random.randint(minval, maxval + 1, dimension, dtype=np.int16)
      return cls(charges=charges)

  return ModularCharge


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
  for n, ct in enumerate(charge_types):
    comb_charges[n] = ct.fuse(charges_A[:, n], charges_B[:, n])
  return np.stack(comb_charges, axis=1)



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


def charge_equal(c1: BaseCharge, c2: BaseCharge) -> bool:
  """
  Compare two BaseCharges `c1` and `c2`.
  Return `True` if they are equal, else `False`.
  """
  if c1.dim != c2.dim:
    return False
  if not np.all(c1.unique_charges == c2.unique_charges):
    return False
  if not np.all(c1.charge_labels == c2.charge_labels):
    return False
  return True
