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

import copy
import warnings
# pylint: disable=line-too-long
from typing import List, Union, Any, Optional, Tuple, Text, Iterable, Type


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

  def copy(self):
    """
    Return a copy of `BaseCharge`.
    """
    obj = self.__new__(type(self))
    obj.__init__(
        charges=self.unique_charges.copy(),
        charge_labels=self.charge_labels.copy(),
        charge_types=self.charge_types)
    return obj

  @property
  def charges(self):
    """
    Return the actual charges of `BaseCharge` as np.ndarray.
    """
    return self.unique_charges[:, self.charge_labels]

  def __repr__(self):
    return str(
        type(self)) + '\n' + 'charges: \n' + self.charges.__repr__() + '\n'

  def __len__(self):
    return len(self.charge_labels)
