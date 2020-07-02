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
import itertools
from typing import List, Optional, Type, Any, Union


def flatten(list_of_list: List[List]) -> List:
  return [l for sublist in list_of_list for l in sublist]



def get_dtype(nbits):
  final_dtype = np.int8
  if nbits > 8:
    final_dtype = np.int16
  if nbits > 16:
    final_dtype = np.int32
  if nbits > 32:
    final_dtype = np.int64
  return final_dtype

def get_resulting_dtype_and_itemsizes(dtypes: List[List[Type[np.number]]]):
  itemsizes =  get_itemsizes(dtypes)

  nbits = sum(flatten(itemsizes))
  if nbits > 64:
    raise TypeError(f"resulting dtype after collapsing {dtypes} "
                    f"has more than 64 bits")
  return get_dtype(nbits), itemsizes

def get_resulting_dtype_and_shifts(dtypes: List[List[Type[np.number]]]):
  itemsizes =  get_itemsizes(dtypes)

  nbits = sum(flatten(itemsizes))
  if nbits > 64:
    raise TypeError(f"resulting dtype after collapsing {dtypes} "
                    f"has more than 64 bits")
  return get_dtype(nbits), get_shifts(itemsizes)

def get_itemsizes(dtypes: List[List[Type[np.number]]]):
  return [[d.itemsize * 8  for d in sublist] for sublist in dtypes]

def get_shifts(itemsizes: List[List[int]]):
  tmp = [sum(l) for l in itemsizes]
  return list(itertools.accumulate((tmp[1:]+[0])[::-1]))[::-1]

def uncollapse_single(charge: np.ndarray, dtypes: List[Type[np.number]]):
  itemsizes = [d.itemsize * 8 for d in dtypes]
  res = [np.bitwise_and(charge, 2**itemsizes[-1] - 1).astype(dtypes[-1])]
  for n in reversed(range(1, len(dtypes))):
    tmp = np.right_shift(charge - res[-1], itemsizes[n])
    res.append(
        np.bitwise_and(tmp, 2**itemsizes[n - 1] - 1).astype(dtypes[n - 1]))
    charge = tmp
  return res[::-1]


def uncollapse(charge: np.ndarray,
               original_dtypes: List[List[Type[np.number]]]):
  itemsizes = [sum(v) for v in get_itemsizes(original_dtypes)]
  res = []
  for n in reversed(range(len(itemsizes))):
    dtype = get_dtype(itemsizes[n])
    shift = np.dtype(dtype).itemsize * 8 - itemsizes[n]
    tmp = np.right_shift(np.left_shift(np.bitwise_and(charge, 2**itemsizes[n] - 1).astype(dtype),shift),shift)
    res.append(tmp)
    charge = np.right_shift(charge - res[-1].astype(charge.dtype), itemsizes[n])
  return res[::-1]


def collapse(charges: List[np.ndarray],
             original_dtypes: List[List[Type[np.number]]]):
  """
  Collapse all `charges` into a single np.ndarray. `original_dtypes`
  contains the original dtypes of each charge in `charges` (i.e. the
  elements of `charges` can themselves be results of collapsing).
  """
  final_dtype, itemsizes = get_resulting_dtype_and_itemsizes(original_dtypes)

  shifts = get_shifts(itemsizes)
  result = np.left_shift(charges[0].astype(final_dtype),
                         shifts[0]).astype(final_dtype)
  for m in range(1, len(charges)):
    result += np.left_shift(charges[m].astype(final_dtype),
                            shifts[m]).astype(final_dtype)
  return result


class BaseCharge:
  """
    Initialize a BaseCharge object.
    Charges of same type are collapsed into a single np.ndarray using bit-shifting.
    Collapsing beyond this is not useful because adding charges of different types
    cannot be performed on collapsed charges.
    Certain functions, like unique and intersect, collapse charges of different types
    to improve performance.
    Args:
      charges: A list of np.ndarray representing (possibly colllapsed) charges.
      charge_types: A list of types of charges.
      num_symmetries: The number of syyemtries
      original_dtypes: A list of list of dtypes for each (possibly collapsed) charge
        in `charges`.
  """

  def __init__(self,
               charges: List[np.ndarray],
               charge_types: List[List[Type["BaseCharge"]]],
               num_symmetries: Optional[int] = None,
               original_dtypes: Optional[List[List]] = None,
               charge_indices: Optional[List[List]] = None) -> None:
    for cts in charge_types:
      if not all([cts[0] is ct for ct in cts]):
        raise ValueError("initialization from completely "
                         "collapsed charges is not allowed")
    self.charge_types = charge_types
    self.stacked_charges = np.stack(charges, axis=0)
    self.charges = charges
    if original_dtypes is None:
      self.original_dtypes = [[c.dtype] for c in charges]
    else:
      self.original_dtypes = original_dtypes
    if charge_indices is None:
      idxs = list(range(len(flatten(self.original_dtypes))))
      pos = [0] + list(
          itertools.accumulate([len(dt) for dt in self.original_dtypes]))
      self.charge_indices = [
          idxs[pos[n - 1]:pos[n]] for n in range(1, len(pos))
      ]
    else:
      self.charge_indices = charge_indices

    # if not np.all([charges[0].shape[0] == c.shape[0] for c in charges]):
    #   raise ValueError("all charges need to have same length.")
    if num_symmetries is None:
      self.num_symmetries = len(self.charges)
    else:
      self.num_symmetries = num_symmetries

    # always collapse charges by default
    self.collapse_charge_types()

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
    return len(self.charges[0])

  def copy(self):
    """
    Return a copy of `BaseCharge`.
    """
    obj = self.__new__(type(self))
    obj.__init__(
        charges=[c.copy() for c in self.charges],
        charge_types=self.charge_types,
        num_symmetries=self.num_symmetries,
        original_dtypes=self.original_dtypes)
    return obj

  @property
  def dtypes(self):
    return [c.dtype for c in self.charges]

  # @property
  # def degeneracies(self) -> np.ndarray:
  #   return np.array(
  #       [np.unique(c, return_counts=True)[1] for c in self._charges])

  def __iter__(self):
    return iter(self.charges)

  def __repr__(self):
    out = "BaseCharge: \n  " + "charge-types: " + str(
        self.charge_types) + "\n  dtypes: " + str(
            self.original_dtypes) + " \n  indices: " + str(
                self.charge_indices
            ) + "\n  " + 'charges: ' + self.charges.__repr__().replace(
                '\n', '\n          ') + '\n'
    return out

  def __len__(self) -> int:
    return self.dim

  # def __eq__(self,
  #            target_charges: Union[np.ndarray, "BaseCharge"]) -> np.ndarray:
  #   if isinstance(target_charges, type(self)):
  #     target_charges = target_charges.charges
  #   if target_charges.ndim > 1:
  #     if target_charges.shape[0] == 1:
  #       target_charges = np.squeeze(target_charges, 0)
  #     else:
  #       raise ValueError("__eq__ only works for 1d arrays")
  #   if len(target_charges) != len(self._charges):
  #     raise ValueError(
  #         "len(target_charges) = {} different from len(self._charges) = {}"
  #         .format(len(target_charges), len(self._charges)))

  #   return np.logical_and.reduce(
  #       [c == target_charges[n] for n, c in enumerate(self._charges)])

  # def __eq__(self,
  #            target_charges: "BaseCharge") -> np.ndarray:
  #   if not isinstance(target_charges, type(self)):
  #     raise TypeError(f"__eq__ only operates on `BaseCharge` objects."
  #                     f" Found {type(target_charges)}")
  #   targets = target_charges.charges
  #   if targets.ndim > 1:
  #     if targets.shape[0] == 1:
  #       target_charges = np.squeeze(target_charges, 0)
  #     else:
  #       raise ValueError("__eq__ only works for 1d arrays")
  #   if len(target_charges) != len(self._charges):
  #     raise ValueError(
  #         "len(target_charges) = {} different from len(self._charges) = {}"
  #         .format(len(target_charges), len(self._charges)))

  #   return np.logical_and.reduce(
  #       [c == target_charges[n] for n, c in enumerate(self._charges)])

  # @property
  # def identity_charges(self) -> np.ndarray:
  #   """
  #   Returns the identity charge.
  #   Returns:
  #     BaseCharge: The identity charge.
  #   """
  #   ids = []
  #   for n, ct in enumerate(self.charge_types):
  #     ids.append(
  #         collapse_charges([ct.identity_charge()] * len(self.original_dtypes[n]),
  #                  self.original_dtypes[n]))
  #   return np.expand_dims(np.stack(ids), 0)

  def collapse_charge_types(self):
    """
    Collapse charges of the same type.
    """
    for cts in self.charge_types:
      unique = set(cts)
      if len(unique) > 1:
        raise ValueError("some lists in charge_types do not"
                         " contain identical values. "
                         "Call uncollapse all first.")

    unique_charge_types = set(flatten(self.charge_types))
    check = []
    for u in unique_charge_types:
      # all values in `ct` are the same
      tmp = [u in ct for ct in self.charge_types]
      check.append(tmp.count(True)==1)
    if all(check):
      #nothing to collapse
      return self

    charge_dict = {}
    charge_dtypes = {}
    charge_indices = {}
    for n, cts in enumerate(self.charge_types):
      ct = cts[0]
      if ct in charge_dict:
        charge_dict[ct].append(self.charges[n])
        charge_dtypes[ct].append(self.original_dtypes[n])
        charge_indices[ct].append(self.charge_indices[n])
      else:
        charge_dict[ct] = [self.charges[n]]
        # these comprehension ARE necessary
        # pylint: disable  = unnecessary-comprehension
        charge_dtypes[ct] = [self.original_dtypes[n]]
        # pylint: disable  = unnecessary-comprehension
        charge_indices[ct] = [self.charge_indices[n]]

    self.charges = []
    self.charge_types = []
    self.original_dtypes = []
    self.charge_indices = []
    for ct, v in charge_dict.items():
      dtypes = charge_dtypes[ct]
      if len(v) > 1:
        #this charge is already collapsed
        collapsed = collapse(v, dtypes)
      else:
        collapsed = v[0]
      self.charges.append(collapsed)
      self.charge_types.append([ct] * len(flatten(dtypes)))
      self.original_dtypes.append(flatten(dtypes))
      self.charge_indices.append(flatten(charge_indices[ct]))

    self.stacked_charges = np.stack(self.charges, axis=0)
    return self

  def uncollapse_charge_types(self):
    # for cts in self.charge_types:
    #   unique = set(cts)
    #   if len(cts) != len(unique):
    #     raise ValueError("some lists in charge_types do not"
    #                      " contain identical values. "
    #                      "Call uncollapse all first.")

    # unique_charge_types = set(flatten(self.charge_types))
    # if len(self.charge_types) == 1:
    #   if len(self.charge_types[0]) != len(unique_charge_types):
    #     raise ValueError("Cannot uncollapse completely collapsed charges"
    #                      " of different types. Use uncollapse_all_charges.")

    if len(self.charges) == len(flatten(self.original_dtypes)):
      return self
    L = sum([len(dt) for dt in self.original_dtypes])
    original_dtypes = [None] * L
    charge_types = [None] * L
    charges = [None] * L
    for n, cts in enumerate(self.charge_types):
      uncollapsed = uncollapse_single(self.charges[n], self.original_dtypes[n])
      for k, m in enumerate(self.charge_indices[n]):
        charges[m] = uncollapsed[k]
        original_dtypes[m] = [self.original_dtypes[n][k]]
        charge_types[m] = cts[k]
    self.charges = charges
    self.original_dtypes = original_dtypes
    self.charge_types = charge_types
    self.charge_indices = [[n] for n in range(len(self.charges))]
    self.stacked_charges = np.stack(self.charges, axis=0)

    return self

  def collapse_all_charges(self):
    """
    Collapse all charges into a single np.ndarray
    """
    if len(self.charges) == 1:
      #nothing to collapse
      return self
    self.charges = [collapse(self.charges, self.original_dtypes)]
    self.charge_types = [flatten(self.charge_types)]
    self.original_dtypes = [flatten(self.original_dtypes)]
    self.charge_indices = [flatten(self.charge_indices)]

  def uncollapse_all_charges(self):
    if len(self.charges) == len(flatten(self.original_dtypes)):
      return self
    L = sum([len(dt) for dt in self.original_dtypes])
    original_dtypes = [None] * L
    charge_types = [None] * L
    charges = [None] * L
    for n, cts in enumerate(self.charge_types):
      uncollapsed = uncollapse_single(self.charges[n], self.original_dtypes[n])
      for k, m in enumerate(self.charge_indices[n]):
        charges[m] = uncollapsed[k]
        original_dtypes[m] = [self.original_dtypes[n][k]]
        charge_types[m] = cts[k]
    self.charges = charges
    self.original_dtypes = original_dtypes
    self.charge_types = charge_types
    self.charge_indices = [[n] for n in range(len(self.charges))]
    self.stacked_charges = np.stack(self.charges, axis=0)

    return self


  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `BaseCharge` object.
    Returns:
      BaseCharge: The result of fusing `self` with `other`.
    """
    # fuse the unique charges from each index, then compute new unique charges
    comb_charges = fuse_ndarray_charges(self._charges, other._charges,
                                        self.charge_types)
    obj = self.__new__(type(self))
    obj.__init__(
        comb_charges,
        self.charge_types,
        self.num_symmetries,
        original_dtypes=self.original_dtypes)

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

      charges = [
          ct.dual_charges(self._charges[n])
          for n, ct in enumerate(self.charge_types)
      ]
      obj = self.__new__(type(self))
      obj.__init__(
          charges,
          self.charge_types,
          self.num_symmetries,
          original_dtypes=self.original_dtypes)
      return obj
    return self

  def __matmul__(self, other):
    if len(self) != len(other):
      raise ValueError(
          '__matmul__ requires charges to have the same number of elements')

    charges = self.charges + other.charges
    charge_types = self.charge_types + other.charge_types
    num_symmetries = self.num_symmetries + other.num_symmetries
    original_dtypes = self.original_dtypes + other.original_dtypes
    R = len(flatten(self.charge_indices))
    charge_indices = self.charge_indices + [[n + R for n in sublist] for sublist in other.charge_indices ]
    return BaseCharge(
        charges=charges,
        charge_types=charge_types,
        num_symmetries=num_symmetries,
        original_dtypes=original_dtypes, charge_indices=charge_indices)

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
        np.ndarray: The indices of the first occurrences of the common values in `self`.
        np.ndarray: The indices of the first occurrences of the common values in `other`.
      If `return_indices=False`:
        BaseCharge
    """
    #TODO: add checks
    len_charges = len(self._charges)
    self.collapse_charge_types()
    collapsed = False
    if len(self._charges) == len_charges:
      collapsed = True

    charge1, dtypes, itemsizes = collapse_charges(self._charges,
                                                  self.original_dtypes)

    other.collapse_charge_types()
    charge2, _, _ = collapse_charges(other._charges, other.original_dtypes)

    tmp = np.intersect1d(
        charge1,
        charge2,
        assume_unique=assume_unique,
        return_indices=return_indices)
    if return_indices:
      final_charges = uncollapse_charges(tmp[0], dtypes, itemsizes)
    else:
      final_charges = uncollapse_charges(tmp, dtypes, itemsizes)
    obj = self.__new__(type(self))
    obj.__init__(
        charges=final_charges,
        charge_types=self.charge_types,
        num_symmetries=self.num_symmetries,
        original_dtypes=self.original_dtypes)
    if not collapsed:
      obj.uncollapse_charge_types()
      self.uncollapse_charge_types()

    if return_indices:
      return obj, tmp[1], tmp[2]

    return obj

  def unique(self,
             return_index=False,
             return_inverse=False,
             return_counts=False) -> Any:
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
    len_charges = len(self._charges)
    if len(self._charges) > 1:
      self.collapse_charge_types()
    collapsed_charge_types = len(self._charges) != len_charges
    collapsed = False
    if len(self._charges) > 1:
      collapsed = True
      charge, dtypes, itemsizes = collapse_charges(self._charges,
                                                   self.original_dtypes)
    charge = self._charges[0]
    tmp = np.unique(charge, return_index, return_inverse, return_counts)
    if return_index or return_inverse or return_counts:
      unique_charges = tmp[0]
    else:
      unique_charges = tmp
    if collapsed:
      final_charges = uncollapse_charges(unique_charges, dtypes, itemsizes)
    else:
      final_charges = [unique_charges]

    obj = self.__new__(type(self))
    obj.__init__(
        charges=final_charges,
        charge_types=self.charge_types,
        num_symmetries=self.num_symmetries,
        original_dtypes=self.original_dtypes)
    if collapsed_charge_types:
      obj.uncollapse_charge_types()
      self.uncollapse_charge_types()
    if return_index or return_inverse or return_counts:
      out = list(tmp)
      out[0] = obj
      return tuple(out)
    return obj

  def reduce(self,
             target_charges: np.ndarray,
             return_locations: bool = False,
             strides: Optional[int] = None) -> Any:
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
    mask = self.isin(target_charges)
    reduced_charges = [c[mask] for c in self._charges]
    obj = self.__new__(type(self))
    obj.__init__(
        reduced_charges,
        self.charge_types,
        self.num_symmetries,
        original_dtypes=self.original_dtypes)

    if return_locations:
      if strides is not None:
        return obj, np.nonzero(mask)[0] * strides
      return obj, np.nonzero(mask)[0]

    return obj

  def __getitem__(self, n: Union[np.ndarray, int]) -> "BaseCharge":
    """
    Args:
      n: An integer or `np.ndarray`.
    Returns:
      np.ndarrau: The charges at `n`.
    """
    if np.isscalar(n):
      n = np.asarray([n])
    obj = self.__new__(type(self))
    obj.__init__([c[n] for c in self._charges],
                 self.charge_types,
                 self.num_symmetries,
                 original_dtypes=self.original_dtypes)

    return obj

  def isin(self, target_charges: np.ndarray) -> np.ndarray:
    """
    See also np.isin.
    Returns an np.ndarray of `dtype=bool`, with `True` at all linear positions
    where `self` is in `target_charges`, and `False` everywhere else.
    Args:
      target_charges: An np.ndarray 
    Returns:
      np.ndarray: An array of boolean values.
    """
    if len(self._charges) == 1:
      target_charges = np.squeeze(target_charges)
      if target_charges.ndim > 1:
        raise ValueError("target_charges.ndim = {} is incompatible with "
                         "number of effective charges = 1".format(
                             target_charges.ndim))

      return np.isin(self._charges[0], target_charges)
    if target_charges.ndim != 2:
      raise ValueError("target_charges.ndim = {} is incompatible with "
                       "number of effective charges = {}".format(
                           target_charges.ndim, len(self._charges)))

    if target_charges.shape[0] != len(self._charges):
      raise ValueError(
          "target_charges.shape[1] = {} is incompatible with len(self._charges) = {}"
          .format(target_charges.shape[1], len(self._charges)))

    tmp = np.expand_dims(np.stack(self._charges, axis=1), 2) == np.expand_dims(
        target_charges, 0)
    #pylint: disable=no-member
    return np.logical_or.reduce(np.logical_and.reduce(tmp, axis=1), axis=1)


class U1Charge(BaseCharge):

  def __init__(self,
               charges: Union[List[np.ndarray], np.ndarray],
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               num_symmetries: Optional[int] = 1,
               original_dtypes: Optional[List[List]] = None) -> None:

    if isinstance(charges, np.ndarray):
      charges = [charges]
    super().__init__(
        charges,
        charge_types=[[type(self)]],
        num_symmetries=num_symmetries,
        original_dtypes=original_dtypes)

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
  def random(cls, minval: int, maxval: int, dimension: int) -> BaseCharge:
    charges = [np.random.randint(minval, maxval, dimension, dtype=np.int16)]
    return cls(charges=charges)


class Z2Charge(BaseCharge):

  def __init__(self,
               charges: Union[List[np.ndarray], np.ndarray],
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               num_symmetries: Optional[int] = 1,
               original_dtypes: Optional[List[List]] = None) -> None:
    if isinstance(charges, np.ndarray):
      charges = [charges]

    super().__init__(
        charges,
        charge_types=[[type(self)]],
        num_symmetries=num_symmetries,
        original_dtypes=original_dtypes)

  @staticmethod
  def fuse(charge1, charge2) -> np.ndarray:
    return np.bitwise_xor.outer(charge1, charge2).ravel()

  @staticmethod
  def dual_charges(charges) -> np.ndarray:
    return charges

  @staticmethod
  def identity_charge() -> np.ndarray:
    return np.int16(0)

  @classmethod
  def random(cls, dimension: int) -> BaseCharge:
    charges = [np.random.randint(0, 2, dimension, dtype=np.int8)]
    return cls(charges=charges)


def fuse_ndarray_charges(charges_A: List[np.ndarray],
                         charges_B: List[np.ndarray],
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
  return [
      ct.fuse(charges_A[n], charges_B[n]) for n, ct in enumerate(charge_types)
  ]


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
  #pylint: disable=no-else-return
  if A.dtype != B.dtype:
    raise TypeError(
        "dtypes of A and B have to match. Found A.dtype = {} and B.dtype = {}"
        .format(A.dtype, B.dtype))
  if A.ndim != B.ndim:
    raise ValueError("array ndims must match to intersect")
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
      #NOTE: we have to copy here, otherwise intersect1d will be super confused

      out = intersect(
          A.T.copy(),
          B.T.copy(),
          axis=0,
          assume_unique=assume_unique,
          return_indices=return_indices)
      if return_indices:
        return out[0].T, out[1], out[2]
      return out.T

    raise NotImplementedError(
        "intersection can only be performed on first or second axis")

  raise NotImplementedError("intersect is only implemented for 1d or 2d arrays")


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
  return np.reshape(
      np.expand_dims(degen1, 1) * np.expand_dims(degen2, 0),
      len(degen1) * len(degen2))


def fuse_ndarrays(arrays: List[Union[List, np.ndarray]]) -> np.ndarray:
  """
  Fuse all `arrays` by simple kronecker addition.
  Arrays are fused from "right to left", 
  Args:
    arrays: A list of arrays to be fused.
  Returns:
    np.ndarray: The result of fusing `arrays`.
  """
  if len(arrays) == 1:
    return np.array(arrays[0])
  fused_arrays = np.asarray(arrays[0])
  for n in range(1, len(arrays)):
    fused_arrays = np.ravel(np.add.outer(fused_arrays, arrays[n]))
  return fused_arrays


def charge_equal(c1, c2):
  if c1.dim != c2.dim:
    return False
  if not np.all(c1.charges == c2.charges):
    return False
  if not np.all([t1 is t2 for t1, t2 in zip(c1.charge_types, c2.charge_types)]):
    return False
  if not np.all([
      t1 is t2 for t1, t2 in zip(
          flatten(c1.original_dtypes), flatten(c2.original_dtypes))
  ]):
    return False
  if len(c1.original_dtypes) != len(c2.original_dtypes):
    return False
  return True
