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


def test_BaseCharge_copy():
  D = 300
  B = 5
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  Q = BaseCharge(charges=q)
  Qcopy = Q.copy()
  assert Q.charge_labels is not Qcopy.charge_labels
  assert Q.unique_charges is not Qcopy.unique_charges
  np.testing.assert_allclose(Q.charge_labels, Qcopy.charge_labels)
  np.testing.assert_allclose(Q.unique_charges, Qcopy.unique_charges)
