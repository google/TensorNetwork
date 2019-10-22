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


class AbelianIndex:
  """
  An index object for creation of abelian, block-sparse tensors
  `AbelianIndex` is a storage class for storing abelian quantum numbers
  of a tensor index. `AbelianIndex` is a wrapper for a python `dict`
  mapping quantum numbers to integers (the dimension of the block)
  """

  @classmethod
  def fromlist(cls, quantumnumbers, dimensions, flow, label=None):
    if all(map(np.isscalar, quantumnumbers)):
      QNs = list(quantumnumbers)
    elif all(list(map(lambda x: not np.isscalar(x), quantumnumbers))):
      QNs = list(map(np.asarray, quantumnumbers))
    else:
      raise TypeError(
          "TensorIndex.fromlist(cls,dictionary,flow,label=None): quantum numbers have inconsistent types"
      )
    return cls(QNs, dimensions, flow, label)

  @classmethod
  def fromdict(cls, dictionary, flow, label=None):
    if all(map(np.isscalar, dictionary.keys())):
      QNs = list(dictionary.keys())
    elif all(list(map(lambda x: not np.isscalar(x), dictionary.keys()))):
      QNs = list(map(np.asarray, dictionary.keys()))
    else:
      raise TypeError(
          "TensorIndex.fromdict(cls,dictionary,flow,label=None): quantum numbers have inconsistent types"
      )

    return cls(QNs, list(dictionary.values()), flow, label)

  def __init__(self, quantumnumbers, dimensions, flow, label=None):
    if __debug__:
      if len(quantumnumbers) != len(dimensions):
        raise ValueError(
            "TensorIndex.__init__: len(quantumnumbers)!=len(dimensions)")

    try:
      unique = dict(zip(quantumnumbers, dimensions))
    except TypeError:
      unique = dict(zip(map(tuple, quantumnumbers), dimensions))

    if __debug__:
      if len(unique) != len(quantumnumbers):
        warnings.warn(
            "in TensorIndex.__init__: found some duplicate quantum numbers; duplicates have been removed"
        )

    if __debug__:
      try:
        mask = np.asarray(list(map(len, unique.keys()))) == len(
            list(unique.keys())[0])
        if not all(mask):
          raise ValueError(
              "in TensorIndex.__init__: found quantum number keys of differing length {0}\n all quantum number have to have identical length"
              .format(list(map(len, unique.keys()))))
      except TypeError:
        if not all(list(map(np.isscalar, unique.keys()))):
          raise TypeError(
              "in TensorIndex.__init__: found quantum number keys of mixed type. all quantum numbers have to be either integers or iterables"
          )
    self._data = np.array(
        list(zip(map(np.asarray, unique.keys()), dimensions)), dtype=object)

    self._flow = flow
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
    self._flow = np.sign(val)
    return self

  def rename(self, label):
    self.label = label
    return self

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
