import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import BaseCharge


def test_BaseCharge_charges():
  D = 100
  B = 6
  charges = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)

  q1 = BaseCharge(charges)
  np.testing.assert_allclose(q1.charges, charges)


def test_BaseCharge_generic():
  D = 300
  B = 5
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  unique = np.unique(q, axis=1)
  Q = BaseCharge(charges=q)
  assert Q.dim == 300
  assert Q.num_symmetries == 2
  assert Q.num_unique == unique.shape[1]


def test_BaseCharge_len():
  D = 300
  B = 5
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  Q = BaseCharge(charges=q)
  assert len(Q) == 300
