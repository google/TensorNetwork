import pytest
import numpy as np
import time
from tensornetwork.backends import backend_factory
from tensornetwork.matrixproductstates.mpo import FiniteMPO, BaseMPO


def test_base_mpo_init(backend):
  mpo = BaseMPO(tensors=[], backend=backend)
  assert mpo.backend.name == backend


def test_finite_mpo_init(backend):
  tensors = [np.random.rand(1, 5, 2, 2), np.random.rand(5, 1, 2, 2)]
  mpo = FiniteMPO(tensors=tensors, backend=backend, name='test')
  assert mpo.backend.name == backend
  assert mpo.name == 'test'


def test_finite_mpo_raises(backend):
  tensors = [np.random.rand(2, 5, 2, 2), np.random.rand(5, 1, 2, 2)]
  with pytest.raises(ValueError):
    FiniteMPO(tensors=tensors, backend=backend)
  tensors = [np.random.rand(1, 5, 2, 2), np.random.rand(5, 2, 2, 2)]
  with pytest.raises(ValueError):
    FiniteMPO(tensors=tensors, backend=backend)
