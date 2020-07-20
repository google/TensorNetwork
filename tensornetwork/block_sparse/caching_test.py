from tensornetwork.block_sparse.caching import (get_cacher, set_caching_status,
                                                get_caching_status, clear_cache,
                                                enable_caching, disable_caching,
                                                _INSTANTIATED_CACHERS)
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import U1Charge
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
from tensornetwork.ncon_interface import ncon
import numpy as np


def test_get_cacher():
  cacher = get_cacher()
  assert len(_INSTANTIATED_CACHERS) == 1
  assert _INSTANTIATED_CACHERS[0] is cacher


def test_set_caching_status():
  set_caching_status(True)
  cacher = get_cacher()
  assert len(_INSTANTIATED_CACHERS) == 1
  assert _INSTANTIATED_CACHERS[0] is cacher
  assert cacher.do_caching

  set_caching_status(False)
  cacher = get_cacher()
  assert len(_INSTANTIATED_CACHERS) == 1
  assert _INSTANTIATED_CACHERS[0] is cacher
  assert not cacher.do_caching


def test_get_caching_status():
  set_caching_status(True)
  assert get_caching_status()
  set_caching_status(False)
  assert not get_caching_status()


def test_enable_caching():
  enable_caching()
  cacher = get_cacher()
  assert len(_INSTANTIATED_CACHERS) == 1
  assert _INSTANTIATED_CACHERS[0] is cacher
  assert cacher.do_caching


def test_disable_caching():
  disable_caching()
  cacher = get_cacher()
  assert len(_INSTANTIATED_CACHERS) == 1
  assert _INSTANTIATED_CACHERS[0] is cacher
  assert not cacher.do_caching


def test_clear_cache():
  D = 100
  M = 5
  mpsinds = [
      Index(U1Charge(np.random.randint(5, 15, D, dtype=np.int16)), False),
      Index(U1Charge(np.array([0, 1, 2, 3], dtype=np.int16)), False),
      Index(U1Charge(np.random.randint(5, 18, D, dtype=np.int16)), True)
  ]
  mpoinds = [
      Index(U1Charge(np.random.randint(0, 5, M)), False),
      Index(U1Charge(np.random.randint(0, 10, M)), True), mpsinds[1],
      mpsinds[1].flip_flow()
  ]
  Linds = [mpoinds[0].flip_flow(), mpsinds[0].flip_flow(), mpsinds[0]]
  Rinds = [mpoinds[1].flip_flow(), mpsinds[2].flip_flow(), mpsinds[2]]
  mps = BlockSparseTensor.random(mpsinds)
  mpo = BlockSparseTensor.random(mpoinds)
  L = BlockSparseTensor.random(Linds)
  R = BlockSparseTensor.random(Rinds)

  enable_caching()
  ncon([L, mps, mpo, R], [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
       backend='symmetric')
  cacher = get_cacher()
  assert len(cacher.cache) > 0
  clear_cache()
  assert len(cacher.cache) == 0
