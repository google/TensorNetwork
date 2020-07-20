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
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import (fuse_charges, fuse_degeneracies,
                                               BaseCharge, fuse_ndarray_charges,
                                               intersect, charge_equal,
                                               fuse_ndarrays)
from typing import List, Union, Any, Tuple, Optional, Sequence
# currently there is only one global cacher that does caching.
# this could be changed later on to having stacks of cachers,
# i.e. different cache levesl
_INSTANTIATED_CACHERS = []
class Cacher:
  def __init__(self):
    self.cache = {}
    self.do_caching = False
    
  def set_status(self, value):
    self.do_caching = value
    
  def clear_cache(self):
    self.cache = {}
    
  @property
  def is_empty(self):
    return len(self.cache) == 0
  
def get_cacher():
  """
  Return a `Cacher` object which can be used to perform 
  caching of block-data for block-sparse tensor contractions.
  """
  if len(_INSTANTIATED_CACHERS) == 0:
    _INSTANTIATED_CACHERS.append(Cacher())
  return _INSTANTIATED_CACHERS[0]
    
def enable_caching():
  """
  Enable caching of block-data for block-sparse contraction.
  If enabled, all data that is needed to perform binary tensor contractions 
  will be cached in a dictionary for later reuse. 
  Enabling caching can significantly speed tensor contractions,
  but can lead to substantially larger memory footprints.
  In particular if the code uses tensor decompositions like QR, SVD
  eig, eigh or any similar method, enabling caching can cause 
  catastrophic memory clutter, so use caching with great care.

  The user can at any point clear the cache by calling 
  `tn.block_sparse.clear_cache()`.
  """
  get_cacher().set_status(True)
  
def disable_caching():
  """
  Disable caching of block-data for block-sparse tensor contractions. 
  Note that the cache WILL NOT BE CLEARED. 
  Clearing the cache can be achieved by calling
  `tn.block_sparse.clear_cache()`.
  """
  get_cacher().set_status(False)
  

def clear_cache():
  """
  Clear the cache that stores block-data for block-sparse tensor contractions.
  """
  get_cacher().clear_cache()
  
def get_caching_status():
  return get_cacher().do_caching

def set_caching_status(status):
  get_cacher().set_status(status)  
