import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import BaseCharge, intersect, fuse_ndarrays, U1Charge, fuse_degeneracies


def test_BaseCharge_charges():
  D = 100
  B = 6
  np.random.seed(10)
  charges = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)

  q1 = BaseCharge(charges)
  np.testing.assert_allclose(q1.charges, charges)


def test_BaseCharge_generic():
  D = 300
  B = 5
  np.random.seed(10)
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  unique = np.unique(q, axis=1)
  Q = BaseCharge(charges=q)
  assert Q.dim == 300
  assert Q.num_symmetries == 2
  assert Q.num_unique == unique.shape[1]


def test_BaseCharge_len():
  D = 300
  B = 5
  np.random.seed(10)
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  Q = BaseCharge(charges=q)
  assert len(Q) == 300


def test_BaseCharge_copy():
  D = 300
  B = 5
  np.random.seed(10)
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  Q = BaseCharge(charges=q)
  Qcopy = Q.copy()
  assert Q.charge_labels is not Qcopy.charge_labels
  assert Q.unique_charges is not Qcopy.unique_charges
  np.testing.assert_allclose(Q.charge_labels, Qcopy.charge_labels)
  np.testing.assert_allclose(Q.unique_charges, Qcopy.unique_charges)


def test_intersect_1():
  a = np.array([[0, 1, 2], [2, 3, 4]])
  b = np.array([[0, -2, 6], [2, 3, 4]])
  out = intersect(a, b, axis=1)
  np.testing.assert_allclose(np.array([[0], [2]]), out)


def test_intersect_2():
  a = np.array([[0, 1, 2], [2, 3, 4]])
  b = np.array([[0, -2, 6, 2], [2, 3, 4, 4]])
  out, la, lb = intersect(a, b, axis=1, return_indices=True)
  np.testing.assert_allclose(np.array([[0, 2], [2, 4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])


def test_intersect_3():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out = intersect(a, b)
  np.testing.assert_allclose([0, 4], out)


def test_intersect_4():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out, la, lb = intersect(a, b, return_indices=True)
  np.testing.assert_allclose([0, 4], out)
  np.testing.assert_allclose(la, [0, 4])
  np.testing.assert_allclose(lb, [0, 2])


def test_fuse_degeneracies():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused_degeneracies = fuse_degeneracies(d1, d2)
  np.testing.assert_allclose(fused_degeneracies, np.kron(d1, d2))


def test_U1Charge_charges():
  D = 100
  B = 6
  np.random.seed(10)
  charges = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

  q1 = U1Charge(charges)
  assert np.all(q1.charges == charges)


def test_U1Charge_dual():
  D = 100
  B = 6
  np.random.seed(10)
  charges = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

  q1 = U1Charge(charges)
  assert np.all(q1.dual(True).charges == -charges)


def get_charges(B0, B1, D, num_charges):
  return [
      np.random.randint(B0, B1 + 1, D).astype(np.int16)
      for _ in range(num_charges)
  ]


def fuse_charges(num_charges, num_charge_types, seed, D, B, use_flows=False):
  np.random.seed(seed)
  if use_flows:
    flows = np.random.choice([True, False], num_charges, replace=True)
  else:
    flows = np.asarray([False] * num_charges)
  np_flows = np.ones(num_charges, dtype=np.int16)
  np_flows[flows] = -1
  charges = [
      get_charges(-B // 2, B // 2, D, num_charge_types)
      for _ in range(num_charges)
  ]
  fused = [
      fuse_ndarrays([charges[n][m] * np_flows[n]
                     for n in range(num_charges)])
      for m in range(num_charge_types)
  ]
  final_charges = [U1Charge(charges[n][0]) for n in range(num_charges)]
  for n in range(num_charges):
    for m in range(1, num_charge_types):
      final_charges[n] = final_charges[n] @ U1Charge(charges[n][m])
  np_target_charges = np.random.randint(-B, B, num_charge_types, dtype=np.int16)
  target_charges = [
      U1Charge(np.array([np_target_charges[n]]))
      for n in range(num_charge_types)
  ]
  target = target_charges[0]
  for m in range(1, num_charge_types):
    target = target @ target_charges[m]
  final = final_charges[0] * flows[0]
  for n in range(1, num_charges):
    final = final + final_charges[n] * flows[n]

  nz_1 = np.nonzero(final == target)[0]
  masks = [fused[m] == target.charges[m, 0] for m in range(num_charge_types)]

  #pylint: disable=no-member
  nz_2 = np.nonzero(np.logical_and.reduce(masks))[0]
  return nz_1, nz_2


@pytest.mark.parametrize('use_flows', [True, False])
@pytest.mark.parametrize('num_charges, num_charge_types, D, B',
                         [(2, 1, 1000, 6), (2, 2, 1000, 6), (3, 1, 100, 6),
                          (3, 2, 100, 6), (3, 3, 100, 6)])
def test_U1Charge_fusion(num_charges, num_charge_types, D, B, use_flows):
  nz_1, nz_2 = fuse_charges(
      num_charges=num_charges,
      num_charge_types=num_charge_types,
      seed=20,
      D=D,
      B=B,
      use_flows=use_flows)
  assert len(nz_1) > 0
  assert len(nz_2) > 0
  assert np.all(nz_1 == nz_2)


def test_BaseCharge_intersect():
  q1 = np.array([[0, 1, 2, 0, 6], [2, 3, 4, -1, 4]])
  q2 = np.array([[0, -2, 6], [2, 3, 4]])
  Q1 = BaseCharge(charges=q1)
  Q2 = BaseCharge(charges=q2)
  res = Q1.intersect(Q2)
  np.testing.assert_allclose(res.charges, np.asarray([[0, 6], [2, 4]]))


def test_BaseCharge_intersect_return_indices():
  q1 = np.array([[0, 1, 2, 0, 6], [2, 3, 4, -1, 4]])
  q2 = np.array([[-2, 0, 6], [3, 2, 4]])
  Q1 = BaseCharge(charges=q1)
  Q2 = BaseCharge(charges=q2)
  res, i1, i2 = Q1.intersect(Q2, return_indices=True)
  #res, i1, i2 = intersect(q1, q2, axis=1, return_indices=True)
  np.testing.assert_allclose(res.charges, np.asarray([[0, 6], [2, 4]]))
  np.testing.assert_allclose(i1, [0, 4])
  np.testing.assert_allclose(i2, [1, 2])


def test_U1Charge_matmul():
  D = 1000
  B = 5
  C1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  C2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  C3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

  q1 = U1Charge(C1)
  q2 = U1Charge(C2)
  q3 = U1Charge(C3)

  Q = q1 @ q2 @ q3
  Q_ = BaseCharge(
      np.stack([C1, C2, C3], axis=0),
      charge_labels=None,
      charge_types=[U1Charge, U1Charge, U1Charge])
  assert np.all(Q.charges == Q_.charges)
