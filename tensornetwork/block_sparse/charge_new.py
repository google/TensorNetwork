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
from typing import List, Optional, Type, Any, Union, Callable
_CACHED_ZNCHARGES = {}
LABEL_DTYPE = np.int16


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
  itemsizes = get_itemsizes(dtypes)

  nbits = sum(flatten(itemsizes))
  if nbits > 64:
    raise TypeError(f"resulting dtype after collapsing {dtypes} "
                    f"has more than 64 bits")
  return get_dtype(nbits), itemsizes


def get_resulting_dtype_and_shifts(dtypes: List[List[Type[np.number]]]):
  itemsizes = get_itemsizes(dtypes)

  nbits = sum(flatten(itemsizes))
  if nbits > 64:
    raise TypeError(f"resulting dtype after collapsing {dtypes} "
                    f"has more than 64 bits")
  return get_dtype(nbits), get_shifts(itemsizes)


def get_itemsizes(dtypes: List[List[Type[np.number]]]):
  return [[d.itemsize * 8 for d in sublist] for sublist in dtypes]


def get_shifts(itemsizes: List[List[int]]):
  tmp = [sum(l) for l in itemsizes]
  return list(itertools.accumulate((tmp[1:] + [0])[::-1]))[::-1]


def expand_single(charge: np.ndarray, dtypes: List[Type[np.number]]):
  itemsizes = [d.itemsize * 8 for d in dtypes]
  res = [np.bitwise_and(charge, 2**itemsizes[-1] - 1).astype(dtypes[-1])]
  for n in reversed(range(1, len(dtypes))):
    # itemsizes always coincide with a fundamental dtype, so no hacky
    # solution neccesary
    tmp = np.right_shift(charge - res[-1], itemsizes[n])
    res.append(
        np.bitwise_and(tmp, 2**itemsizes[n - 1] - 1).astype(dtypes[n - 1]))
    charge = tmp
  return res[::-1]


def expand(charge: np.ndarray,
           original_dtypes: List[List[Type[np.number]]]) -> List[np.ndarray]:
  itemsizes = [sum(v) for v in get_itemsizes(original_dtypes)]
  res = []
  for n in reversed(range(len(itemsizes))):
    dtype = get_dtype(itemsizes[n])
    shift = np.dtype(dtype).itemsize * 8 - itemsizes[n]
    # a little hacky solution to fill 1s from the left side
    # into the bitrep of the masked number if neccesary
    # numpy uses two's complement representation of negative ints,
    # this restores the representation of charges with itemsizes
    # which don't match the itemsize of its fundamental dtype
    # (e.g. itemsize 24 with fundamental itemsize 32 for int32)
    res.append(
        np.right_shift(
            np.left_shift(
                np.bitwise_and(charge, 2**itemsizes[n] - 1).astype(dtype),
                shift), shift))
    charge = np.right_shift(charge - res[-1].astype(charge.dtype), itemsizes[n])
  return res[::-1]


def collapse(charges: List[np.ndarray],
             original_dtypes: List[List[Type[np.number]]]) -> np.ndarray:
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
    Charges of same type are collapsed into a single np.ndarray using 
    bit-shifting. Collapsing beyond this is not useful because adding
    charges of different types cannot be performed on collapsed charges.
    Certain functions, like unique and intersect, collapse charges of
    different types to improve performance.
    Args:
      charges: A list of np.ndarray representing (possibly colllapsed)
        charges.
      charge_types: A list of types of charges.
      original_dtypes: A list of list of dtypes for each
        (possibly collapsed) charge in `charges`.
  """

  def __init__(self,
               charges: Union[np.ndarray, List[np.ndarray]],
               charge_types: Optional[List[List[Type["BaseCharge"]]]] = None,
               original_dtypes: Optional[List[List]] = None,
               charge_indices: Optional[List[List]] = None) -> None:
    
    if not isinstance(charges, list):
      charges = [np.array(charges)]

    self.charges = charges

    if charge_types is None:
      self.charge_types = [[type(self)] for _ in range(len(self.charges))]
    else:
      for n, cts in enumerate(charge_types):
        if not all([cts[0] is ct for ct in cts]):
          raise ValueError("Not all charge-types in `charge_types[{n}]` "
                           "are the same, found {cts}.")
    if charge_types is None:
      charge_types = [[type(self)] for _ in range(len(self.charges))]
    self.charge_types = charge_types
    if original_dtypes is None:
      self.original_dtypes = [[c.dtype] for c in self.charges]
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

    # always collapse charge-types by default
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
  def random(cls, dimension: int, minval: int, maxval: int):
    raise NotImplementedError(
        "`random` has to be implemented in derived classes")

  @property
  def num_symmetries(self):
    return len(flatten(self.charge_types))

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
        original_dtypes=self.original_dtypes,
        charge_indices=self.charge_indices)
    return obj

  @property
  def dtypes(self):
    return [c.dtype for c in self.charges]

  @property
  def is_collapsed(self):
    return len(self.charges) == 1

  def __iter__(self):
    return iter(self.charges)

  @property
  def names(self):
    return repr([[ct.__new__(ct).__class__.__name__
                  for ct in cts]
                 for cts in self.charge_types])

  def __repr__(self):
    dtype_names = repr([
        [np.dtype(dt).name for dt in dtypes] for dtypes in self.original_dtypes
    ])
    tmp = ''.join([str(c) + '\n ' for c in self.charges
                  ]).replace('\n', '\n          ')
    out = "BaseCharge: \n  charge-types: " + self.names + \
      "\n  dtypes: " + dtype_names + " \n  indices: " + \
      str(self.charge_indices) + "\n  charges: " + tmp

    return out

  def __len__(self) -> int:
    return self.dim

  def __eq__(self, other: "BaseCharge") -> np.ndarray:
    # collapse into a single nparray
    self_is_collapsed = self.is_collapsed
    if not self_is_collapsed:
      self.collapse()
    other_is_collapsed = other.is_collapsed
    if not other_is_collapsed:
      other.collapse()

    res = self.charges[0][:, None] == other.charges[0][None, :]
    return np.squeeze(res)

  def identity_charges(self, dim: int = 1) -> "BaseCharge":
    """
    Returns the identity charge.
    Returns:
      BaseCharge: The identity charge.
    """
    charges = []
    for n, cts in enumerate(self.charge_types):
      tmpcharges = []
      for ct in cts:
        iden = ct.identity_charge()
        tmpcharges.append(np.full(dim, fill_value=iden, dtype=iden.dtype))
      dtypes = [[dt] for dt in self.original_dtypes[n]]
      charges.append(collapse(tmpcharges, dtypes))
    is_collapsed = self.is_collapsed

    obj = self.__new__(type(self))
    obj.__init__(
        charges=charges,
        charge_types=self.charge_types,
        original_dtypes=self.original_dtypes,
        charge_indices=self.charge_indices)
    if is_collapsed:
      obj.collapse()
    return obj

  def collapse_charge_types(self):
    """
    Collapse charges of sames types into a single np.ndarray.
    """
    for cts in self.charge_types:
      unique = set(cts)
      if len(unique) > 1:
        raise ValueError("some lists in charge_types do not"
                         " contain identical values. "
                         "Call expand all first.")

    unique_charge_types = set(flatten(self.charge_types))
    check = []
    for u in unique_charge_types:
      # all values in `ct` are the same
      tmp = [u in ct for ct in self.charge_types]
      check.append(tmp.count(True) == 1)
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

    return self

  def expand_charge_types(self):
    """
    Expand collapsed charges of same types into one np.ndarray
    for each charge. This is the reverse operation of 
    `collapse_charge_types`.
    """
    if len(self.charges) != len(self.charge_types):
      raise ValueError("calling `expand_charge_types` on a "
                       "collapsed BaseCharge with "
                       "with more than one charge-types is "
                       "not possible. Try calling `expand` first.")

    if len(self.charges) == len(flatten(self.original_dtypes)):
      return self
    L = sum([len(dt) for dt in self.original_dtypes])
    original_dtypes = [None] * L
    charge_types = [None] * L
    charges = [None] * L
    for n, cts in enumerate(self.charge_types):
      expanded = expand_single(self.charges[n], self.original_dtypes[n])
      for k, m in enumerate(self.charge_indices[n]):
        charges[m] = expanded[k]
        original_dtypes[m] = [self.original_dtypes[n][k]]
        charge_types[m] = [cts[k]]
    self.charges = charges
    self.original_dtypes = original_dtypes
    self.charge_types = charge_types
    self.charge_indices = [[n] for n in range(len(self.charges))]

    return self

  def collapse(self):
    """
    Collapse all charges into a single np.ndarray.
    """
    if len(self.charges) == 1:
      #nothing to collapse
      return self
    self.charges = [collapse(self.charges, self.original_dtypes)]
    return self

  def expand(self):
    """
    Expand charges. This is the reverse operation to `collapse`.
    `expand` will restore the state prior to the last `collapse` call.
    If `collapse` has not been called, `expand` has not effect.
    """
    if len(self.charges) == len(self.original_dtypes):
      # nothing to expand
      return self
    if len(self.charges) == 1:
      self.charges = expand(self.charges[0], self.original_dtypes)
      return self
    raise ValueError(f"Found inconsistent BaseCharge object."
                     f"len(BaseCharge.charges)  = {len(self.charges)}"
                     f" is different from 1 and differs from "
                     f"len(BaseCharge.charge_types) = "
                     f"{len(self.charge_types)}")

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `BaseCharge` object.
    Returns:
      BaseCharge: The result of fusing `self` with `other`.
    """
    # fuse the unique charges from each index,
    # then compute new unique charges
    # Note (mganahl): check if all cts are identical is
    #                 performed below in __init__
    self.expand()
    other.expand()

    if len(self.charges) == 1 and len(self.charge_types) > 1:
      raise ValueError("self is collapsed: cannot add collapsed charges")
    if len(other.charges) == 1 and len(other.charge_types) > 1:
      raise ValueError("other is collapsed: cannot add collapsed charges")
    charge_types = [cts[0] for cts in self.charge_types]
    fused_charges = fuse_ndarray_charges(self.charges, other.charges,
                                         charge_types)
    # self.collapse()
    # other.collapse()

    obj = self.__new__(type(self))
    obj.__init__(
        charges=fused_charges,
        charge_types=self.charge_types,
        original_dtypes=self.original_dtypes,
        charge_indices=self.charge_indices)
    # obj.collapse()
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
      is_collapsed = self.is_collapsed
      if is_collapsed:
        self.expand()
      # we need to expand identical types as well,
      # dual operation does in general not respect bit shifting
      self.expand_charge_types()
      charges = [
          self.charge_types[n][0].dual_charges(c)
          for n, c in enumerate(self.charges)
      ]
      obj = self.__new__(type(self))
      obj.__init__(
          charges=charges,
          charge_types=self.charge_types,
          original_dtypes=self.original_dtypes,
          charge_indices=self.charge_indices)
      return obj
    return self

  def __matmul__(self, other):
    if len(self) != len(other):
      raise ValueError(
          '__matmul__ requires charges to have the same number of elements')

    charges = self.charges + other.charges
    charge_types = self.charge_types + other.charge_types
    original_dtypes = self.original_dtypes + other.original_dtypes
    R = len(flatten(self.charge_indices))
    charge_indices = self.charge_indices + [[n + R
                                             for n in sublist]
                                            for sublist in other.charge_indices]
    return BaseCharge(
        charges=charges,
        charge_types=charge_types,
        original_dtypes=original_dtypes,
        charge_indices=charge_indices)

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
        np.ndarray: The indices of the first occurrences 
          of the common values in `self`.
        np.ndarray: The indices of the first occurrences 
          of the common values in `other`.
      If `return_indices=False`:
        BaseCharge
    """
    self_is_collapsed = self.is_collapsed
    other_is_collapsed = other.is_collapsed
    if not self_is_collapsed:
      self.collapse()
    if not other_is_collapsed:
      other.collapse()

    res = np.intersect1d(
        self.charges[0],
        other.charges[0],
        assume_unique=assume_unique,
        return_indices=return_indices)
    if return_indices:
      charges = res[0]
    else:
      charges = res
    obj = self.__new__(type(self))
    obj.__init__(
        charges=[charges],
        charge_types=self.charge_types,
        original_dtypes=self.original_dtypes,
        charge_indices=self.charge_indices)

    if return_indices:
      return obj, res[1], res[2]

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
      return_index: If `True`, also return the indices of 
        `self.charges` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
      return_inverse: If `True`, also return the indices of the unique array 
        (for the specified axis, if provided) that can be used to reconstruct 
        `self.charges`.
      return_counts: If `True`, also return the number of times each unique 
        item appears in `self.charges`.
    Returns:
      BaseCharge: The sorted unique values.
      np.ndarray: The indices of the first occurrences of the unique 
        values in the original array. Only provided if `return_index` 
        is True.
      np.ndarray: The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
      np.ndarray: The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.
    """
    is_collapsed = self.is_collapsed
    if not is_collapsed:
      self.collapse()
    res = np.unique(self.charges[0], return_index, return_inverse,
                    return_counts)

    if any([return_index, return_inverse, return_counts]):
      res = list(res)
      charges = res[0]
    else:
      charges = res

    obj = self.__new__(type(self))
    obj.__init__(
        charges=[charges],
        charge_types=self.charge_types,
        original_dtypes=self.original_dtypes,
        charge_indices=self.charge_indices)

    if return_inverse and not return_index:
      res[1] = res[1].astype(LABEL_DTYPE)
    if return_inverse and return_index:
      res[2] = res[2].astype(LABEL_DTYPE)  #always use int16 dtypes for labels
    if any([return_index, return_inverse, return_counts]):
      return [obj] + res[1:]
    return obj

  def reduce(self,
             targets: "BaseCharge",
             return_locations: bool = False,
             return_type: str = 'labels',
             return_unique: bool = True,
             strides: Optional[int] = None) -> Any:
    """
    Reduce the dimension of the charge to keep only the charge 
    values that intersect with target_charges
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
    if not self.is_collapsed:
      self.collapse()
    if not targets.is_collapsed:
      targets.collapse()

    unique, labels = self.unique(return_inverse=True)
    reduced_unique_charges, reduced_unique_labels, _ = unique.intersect(
        targets, return_indices=True)
    mapping = np.full(len(unique), fill_value=-1, dtype=LABEL_DTYPE)
    mapping[reduced_unique_labels] = np.arange(
        len(reduced_unique_labels), dtype=LABEL_DTYPE)
    tmp = mapping[labels]
    reduced_labels = tmp[tmp >= 0]
    if return_type == 'labels':
      res = reduced_labels
    elif return_type == 'charges':
      res = reduced_unique_charges[reduced_labels]
    else:
      raise ValueError(f"unrecognized value {return_type} for return_type."
                       f" Allowed values are 'labels' anbd 'charges'")

    if return_locations:
      locations = np.nonzero(np.isin(labels, reduced_unique_labels))[0]
      if strides is not None:
        locations *= strides
      if return_unique:
        return res, locations, reduced_unique_charges
      return res, locations

    if return_unique:
      return res, reduced_unique_charges
    return res

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
    obj.__init__(
        charges=[c[n] for c in self.charges],
        charge_types=self.charge_types,
        original_dtypes=self.original_dtypes,
        charge_indices=self.charge_indices)

    return obj

  def isin(self, other: "BaseCharge") -> np.ndarray:
    """
    See also np.isin.
    Returns an np.ndarray of `dtype=bool`, with `True` at all linear positions
    where `self` is in `target_charges`, and `False` everywhere else.
    Args:
      target_charges: An np.ndarray 
    Returns:
      np.ndarray: An array of boolean values.
    """
    self_is_collapsed = self.is_collapsed
    other_is_collapsed = other.is_collapsed
    if not self_is_collapsed:
      self.collapse()
    if not other_is_collapsed:
      other.collapse()

    res = np.isin(self.charges[0], other.charges[0])
    return res


class U1Charge(BaseCharge):

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
    charges = [np.random.randint(minval, maxval, dimension, dtype=np.int16)]
    return cls(charges=charges)


class Z2Charge(BaseCharge):

  def __init__(self,
               charges: Union[List[np.ndarray], np.ndarray],
               charge_types: Optional[List[Type["BaseCharge"]]] = None,
               original_dtypes: Optional[List[List]] = None,
               charge_indices: Optional[List[List]] = None) -> None:
    unique = np.unique(np.ravel(charges))
    if not np.all(np.isin(unique, [0, 1])):
      raise ValueError("Z2 charges can only be 0 or 1, found {}".format(unique))

    super().__init__(
        charges=charges,
        charge_types=[[type(self)]],
        original_dtypes=original_dtypes,
        charge_indices=charge_indices)

  @staticmethod
  def fuse(charge1: np.ndarray, charge2: np.ndarray) -> np.ndarray:
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
    charges = [np.random.randint(0, 2, dimension, dtype=np.int8)]
    return cls(charges=charges)


def ZNCharge(n: int) -> Callable:
  """Constructor for charge classes of the ZN symmetry groups.

  Args:
    n: The module of the symmetry group.
  Returns:
    A charge class of your given ZN symmetry group.
  """
  if n < 2:
    raise ValueError(f"n must be >= 2, found {n}")

  class ModularCharge(BaseCharge):

    def __init__(self,
                 charges: Union[List[np.ndarray], np.ndarray],
                 charge_types: Optional[List[Type["BaseCharge"]]] = None,
                 original_dtypes: Optional[List[List]] = None,
                 charge_indices: Optional[List[List]] = None) -> None:
      unique = np.unique(np.ravel(charges))
      if not np.all(np.isin(unique, list(range(n)))):
        raise ValueError(f"Z{n} charges must be in range({n}), found: {unique}")
      super().__init__(
          charges=charges,
          charge_types=[[type(self)]],
          original_dtypes=original_dtypes,
          charge_indices=charge_indices)

    @staticmethod
    def fuse(charge1, charge2) -> np.ndarray:
      #pylint: disable=no-member
      return np.outer(charge1, charge2).ravel() % n

    @staticmethod
    def dual_charges(charges) -> np.ndarray:
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
      charges = np.random.randint(minval, maxval + 1, dimension, dtype=np.int8)
      return cls(charges=charges)
    
  if n not in _CACHED_ZNCHARGES:
    _CACHED_ZNCHARGES[n] = ModularCharge
  return _CACHED_ZNCHARGES[n]


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
  res = True
  if c1.dim != c2.dim:
    res = False
  if res and len(c1.charges) != len(c2.charges):
    res = False
  if res:
    for charge1, charge2 in zip(c1.charges, c2.charges):
      if not np.all(charge1 == charge2):
        res = False
        break
  if res and len(c1.charge_types) != len(c2.charge_types):
    res = False
  if res:
    for ct1, ct2 in zip(c1.charge_types, c2.charge_types):
      if len(ct1) != len(ct2):
        res = False
        break
  if res:
    if not all([
        t1 is t2
        for t1, t2 in zip(flatten(c1.charge_types), flatten(c2.charge_types))
    ]):
      res = False
  if res and len(c1.original_dtypes) != len(c2.original_dtypes):
    res = False
  if res:
    for dt1, dt2 in zip(c1.original_dtypes, c2.original_dtypes):
      if len(dt1) != len(dt2):
        res = False
        break
  if res:
    if not all([
        t1 is t2 for t1, t2 in zip(
            flatten(c1.original_dtypes), flatten(c2.original_dtypes))
    ]):
      res = False

  return res


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

  # see https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays pylint: disable-line-too-long
  # pylint: disable=no-else-return
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
