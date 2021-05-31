import pytest
import numpy as np
import tensorflow as tf
import jax
import torch
from tensornetwork.backends import backend_factory
#pylint: disable=line-too-long
from tensornetwork.matrixproductstates.mpo import (FiniteMPO,
                                                   BaseMPO,
                                                   InfiniteMPO,
                                                   FiniteFreeFermion2D)
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tensornetwork.matrixproductstates.dmrg import FiniteDMRG




@pytest.fixture(
    name="backend_dtype_values",
    params=[('numpy', np.float64), ('numpy', np.complex128),
            ('tensorflow', tf.float64), ('tensorflow', tf.complex128),
            ('pytorch', torch.float64), ('jax', np.float64),
            ('jax', np.complex128)])
def backend_dtype(request):
  return request.param


def test_base_mpo_init(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]
  tensors = [
      backend.randn((1, 5, 2, 2), dtype=dtype),
      backend.randn((5, 5, 2, 2), dtype=dtype),
      backend.randn((5, 1, 2, 2), dtype=dtype)
  ]
  mpo = BaseMPO(tensors=tensors, backend=backend, name='test')
  assert mpo.backend is backend
  assert mpo.dtype == dtype
  np.testing.assert_allclose(mpo.bond_dimensions, [1, 5, 5, 1])


def test_base_mpo_raises():
  backend = backend_factory.get_backend('numpy')
  tensors = [
      backend.randn((1, 5, 2, 2), dtype=np.float64),
      backend.randn((5, 5, 2, 2), dtype=np.float64),
      backend.randn((5, 1, 2, 2), dtype=np.float32)
  ]
  with pytest.raises(TypeError):
    BaseMPO(tensors=tensors, backend=backend)
  mpo = BaseMPO(tensors=[], backend=backend)
  mpo.tensors = tensors
  with pytest.raises(TypeError):
    mpo.dtype


def test_finite_mpo_raises(backend):
  tensors = [np.random.rand(2, 5, 2, 2), np.random.rand(5, 1, 2, 2)]
  with pytest.raises(ValueError):
    FiniteMPO(tensors=tensors, backend=backend)
  tensors = [np.random.rand(1, 5, 2, 2), np.random.rand(5, 2, 2, 2)]
  with pytest.raises(ValueError):
    FiniteMPO(tensors=tensors, backend=backend)


def test_infinite_mpo_raises(backend):
  tensors = [np.random.rand(2, 5, 2, 2), np.random.rand(5, 3, 2, 2)]
  with pytest.raises(ValueError):
    InfiniteMPO(tensors=tensors, backend=backend)


def test_infinite_mpo_roll(backend):
  tensors = [np.random.rand(5, 5, 2, 2), np.random.rand(5, 5, 2, 2)]
  mpo = InfiniteMPO(tensors=tensors, backend=backend)
  mpo.roll(1)
  np.testing.assert_allclose(mpo.tensors[0], tensors[1])
  np.testing.assert_allclose(mpo.tensors[1], tensors[0])
  mpo.roll(1)
  np.testing.assert_allclose(mpo.tensors[0], tensors[0])
  np.testing.assert_allclose(mpo.tensors[1], tensors[1])


def test_len(backend):
  tensors = [
      np.random.rand(1, 5, 2, 2),
      np.random.rand(5, 5, 2, 2),
      np.random.rand(5, 1, 2, 2)
  ]
  mpo = BaseMPO(tensors=tensors, backend=backend)
  assert len(mpo) == 3


@pytest.mark.parametrize("N1, N2, D", [(2, 2, 4), (2, 4, 16), (4, 4, 128)])
def test_finiteFreeFermions2d(N1, N2, D):
  def adjacency(N1, N2):
    neighbors = {}
    mat = np.arange(N1 * N2).reshape(N1, N2)
    for n in range(N1 * N2):
      x, y = np.divmod(n, N2)
      if n not in neighbors:
        neighbors[n] = []
      if y < N2 - 1:
        neighbors[n].append(mat[x, y + 1])
      if x > 0:
        neighbors[n].append(mat[x - 1, y])
    return neighbors

  adj = adjacency(N1, N2)
  tij = np.zeros((N1 * N2, N1 * N2))
  t = -1
  v = -1
  for n, d in adj.items():
    for ind in d:
      tij[n, ind] += t
      tij[ind, n] += t
  tij += np.diag(np.ones(N1 * N2) * v)

  eta, _ = np.linalg.eigh(tij)
  expected = min(np.cumsum(eta))

  t1 = t
  t2 = t
  dtype = np.float64
  mpo = FiniteFreeFermion2D(t1, t2, v, N1, N2, dtype)
  mps = FiniteMPS.random([2] * N1 * N2, [D] * (N1 * N2 - 1), dtype=np.float64)
  dmrg = FiniteDMRG(mps, mpo)
  actual = dmrg.run_one_site()
  np.testing.assert_allclose(actual, expected)
