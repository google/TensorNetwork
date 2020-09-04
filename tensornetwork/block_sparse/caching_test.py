from tensornetwork.block_sparse.caching import (get_cacher, set_caching_status,
                                                get_caching_status, clear_cache,
                                                enable_caching, disable_caching,
                                                _INSTANTIATED_CACHERS)
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import U1Charge, charge_equal
from tensornetwork.block_sparse.blocksparse_utils import (
    _to_string, _find_transposed_diagonal_sparse_blocks)

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
  disable_caching()

def test_disable_caching():
  disable_caching()
  cacher = get_cacher()
  assert len(_INSTANTIATED_CACHERS) == 1
  assert _INSTANTIATED_CACHERS[0] is cacher
  assert not cacher.do_caching


def test_cache():
  D = 10
  mpsinds = [
      Index(U1Charge(np.random.randint(-5, 5, D, dtype=np.int16)), False),
      Index(U1Charge(np.random.randint(-5, 5, D, dtype=np.int16)), False),
      Index(U1Charge(np.random.randint(-5, 5, D, dtype=np.int16)), False),
      Index(U1Charge(np.random.randint(-5, 5, D, dtype=np.int16)), True)
  ]
  A = BlockSparseTensor.random(mpsinds)
  B = A.conj()
  res_charges = [
      A.flat_charges[2], A.flat_charges[3], B.flat_charges[2], B.flat_charges[3]
  ]
  res_flows = [
      A.flat_flows[2], A.flat_flows[3], B.flat_flows[2], B.flat_flows[3]
  ]

  enable_caching()
  ncon([A, B], [[1, 2, -1, -2], [1, 2, -3, -4]], backend='symmetric')
  cacher = get_cacher()
  sA = _to_string(A.flat_charges, A.flat_flows, 2, [2, 3, 0, 1])
  sB = _to_string(B.flat_charges, B.flat_flows, 2, [0, 1, 2, 3])
  sC = _to_string(res_charges, res_flows, 2, [0, 1, 2, 3])
  blocksA, chargesA, dimsA = _find_transposed_diagonal_sparse_blocks(
      A.flat_charges, A.flat_flows, 2, [2, 3, 0, 1])
  blocksB, chargesB, dimsB = _find_transposed_diagonal_sparse_blocks(
      B.flat_charges, B.flat_flows, 2, [0, 1, 2, 3])
  blocksC, chargesC, dimsC = _find_transposed_diagonal_sparse_blocks(
      res_charges, res_flows, 2, [0, 1, 2, 3])

  assert sA in cacher.cache
  assert sB in cacher.cache
  assert sC in cacher.cache

  for b1, b2 in zip(cacher.cache[sA][0], blocksA):
    np.testing.assert_allclose(b1, b2)
  for b1, b2 in zip(cacher.cache[sB][0], blocksB):
    np.testing.assert_allclose(b1, b2)
  for b1, b2 in zip(cacher.cache[sC][0], blocksC):
    np.testing.assert_allclose(b1, b2)
  assert charge_equal(cacher.cache[sA][1], chargesA)
  assert charge_equal(cacher.cache[sB][1], chargesB)
  assert charge_equal(cacher.cache[sC][1], chargesC)

  np.testing.assert_allclose(cacher.cache[sA][2], dimsA)
  np.testing.assert_allclose(cacher.cache[sB][2], dimsB)
  np.testing.assert_allclose(cacher.cache[sC][2], dimsC)
  disable_caching()
  clear_cache()

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
  disable_caching()
  clear_cache()
  assert len(cacher.cache) == 0
