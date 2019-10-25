import collections
import numpy as np
import operator
import warnings
import os
import sys
#import qutilities as qutils
#import utils as cutils
import functools as fct
import copy
from typing import Iterable, Optional, Text


class AbelianIndex:
  """
  An index object for creation of abelian, block-sparse tensors
  `AbelianIndex` is a storage class for storing abelian quantum numbers
  of a tensor index. `AbelianIndex` is a wrapper for a python `dict`
  mapping quantum numbers to integers (the dimension of the block).
  `AbelianIndex` can have a `flow` denoting the "flow of charge".
  """

  @classmethod
  def fromlist(cls,
               quantumnumbers: Iterable,
               dimensions: Iterable[int],
               flow: int,
               label: Optional[Text] = None):
    if all(map(np.isscalar, quantumnumbers)):
      QNs = list(quantumnumbers)
    elif all(list(map(lambda x: not np.isscalar(x), quantumnumbers))):
      QNs = list(map(np.asarray,
                     quantumnumbers))  #turn quantum numbers into np.ndarray
    else:
      raise TypeError("quantum numbers have inconsistent types")
    return cls(QNs, dimensions, flow, label)

  def __init__(self,
               quantumnumbers: Iterable,
               dimensions: Iterable[int],
               flow: int,
               label: Optional[Text] = None):
    try:
      unique = dict(zip(quantumnumbers, dimensions))
    except TypeError:
      unique = dict(zip(map(tuple, quantumnumbers), dimensions))
    if len(unique) != len(quantumnumbers):
      warnings.warn("removing duplicate quantum numbers")
    try:
      lengths = np.asarray([len(k) for k in unique.keys()])
      if not all(lengths == lenghts[0])
        raise ValueError(
            "quantum number have differing lengths")
    except TypeError:
      if not all(list(map(np.isscalar, unique.keys()))):
        raise TypeError(
            "quantum numbers have mixed types")
        )
    self.data = np.array(
        list(zip(map(np.asarray, unique.keys()), dimensions)), dtype=object)

    self.flow = flow
    self.label = label

  def __getitem__(self, n):
    return self._data[n[0], n[1]]

  def Q(self, n):
    return self._data[n, 0]

  def D(self, n):
    return self._data[n, 1]

  def __len__(self):
    return self._data.shape[0]

  def setflow(self, val):
    if val == 0:
      raise ValueError(
          "TensorIndex.flow: trying to set TensorIndex._flow to 0, use positive or negative integers only"
      )
    self.flow = 1 if val > 0 else -1

  def rename(self, label):
    self.label = label

  @property
  def flow(self):
    return self._flow

  @flow.setter
  def flow(self, val):
    if val == 0:
      raise ValueError(
          "TensorIndex.flow: trying to set TensorIndex._flow to 0, use positive or negative integers only"
      )
    self._flow = np.sign(val)

  @property
  def shape(self):
    return self._data.shape

  @property
  def DataFrame(self):
    return pd.DataFrame.from_records(data=self._data, columns=['qn', 'D'])

  def __str__(self):
    print('')
    print('TensorIndex, label={0}, flow={1}'.format(self.label, self.flow))
    print(self.DataFrame)
    return ''
