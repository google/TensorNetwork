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
_INSTANTIATED_CACHERS = []
class Cacher:
  def __init__(self):
    self.cache = {}
    self.do_caching = False
    
  def set_status(self, value):
    self.do_caching = value
    
  def clear_cache(self):
    self.cache = {}
  
def cacher_factory():
  if len(_INSTANTIATED_CACHERS) == 0:
    _INSTANTIATED_CACHERS.append(Cacher())
  return _INSTANTIATED_CACHERS[0]
    
def enable_caching():
  _INSTANTIATED_CACHERS[0].set_status(True)
  
def disable_caching():
  _INSTANTIATED_CACHERS[0].set_status(False)

def clear_cache():
  _INSTANTIATED_CACHERS[0].clear_cache()


  
