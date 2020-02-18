import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import BaseCharge, U1Charge, fuse_degeneracies, fuse_ndarrays


def test_fuse_degeneracies():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused_degeneracies = fuse_degeneracies(d1, d2)
  np.testing.assert_allclose(fused_degeneracies, np.kron(d1, d2))


def test_U1Charge_charges():
  D = 100
  B = 6
  charges = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

  q1 = U1Charge(charges)
  assert np.all(q1.charges == charges)


def test_U1Charge_dual():
  D = 100
  B = 6
  charges = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

  q1 = U1Charge(charges)
  assert np.all(q1.dual(True).charges == -charges)


def test_U1Charge_fusion():

  def run_test():
    D = 2000
    B = 6
    O1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    O2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    P1 = np.random.randint(0, B + 1, D).astype(np.int16)
    P2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    Q1 = np.random.randint(1, B + 1, D).astype(np.int16)
    Q2 = np.random.randint(1, B + 1, D).astype(np.int16)

    charges_1 = [O1, O2]
    charges_2 = [P1, P2]
    charges_3 = [Q1, Q2]

    fused_1 = fuse_ndarrays(charges_1)
    fused_2 = fuse_ndarrays(charges_2)
    fused_3 = fuse_ndarrays(charges_3)
    q1 = U1Charge(O1) @ U1Charge(P1) @ U1Charge(Q1)
    q2 = U1Charge(O2) @ U1Charge(P2) @ U1Charge(Q2)

    target = BaseCharge(
        charges=np.random.randint(-B, B, (3, 1), dtype=np.int16),
        charge_labels=None,
        charge_types=[U1Charge, U1Charge, U1Charge])
    q12 = q1 + q2

    nz_1 = np.nonzero(q12 == target)[0]
    i1 = fused_1 == target.charges[0, 0]
    i2 = fused_2 == target.charges[1, 0]
    i3 = fused_3 == target.charges[2, 0]
    #pylint: disable=no-member
    nz_2 = np.nonzero(np.logical_and.reduce([i1, i2, i3]))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()

  assert np.all(nz_1 == nz_2)


def test_U1Charge_multiple_fusion():

  def run_test():
    D = 300
    B = 4
    O1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    O2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    O3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    P1 = np.random.randint(0, B + 1, D).astype(np.int16)
    P2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    P3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    Q1 = np.random.randint(1, B + 1, D).astype(np.int16)
    Q2 = np.random.randint(0, B + 1, D).astype(np.int16)
    Q3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

    charges_1 = [O1, O2, O3]
    charges_2 = [P1, P2, P3]
    charges_3 = [Q1, Q2, Q3]

    fused_1 = fuse_ndarrays(charges_1)
    fused_2 = fuse_ndarrays(charges_2)
    fused_3 = fuse_ndarrays(charges_3)
    q1 = U1Charge(O1) @ U1Charge(P1) @ U1Charge(Q1)
    q2 = U1Charge(O2) @ U1Charge(P2) @ U1Charge(Q2)
    q3 = U1Charge(O3) @ U1Charge(P3) @ U1Charge(Q3)

    target = BaseCharge(
        charges=np.random.randint(-B, B, (3, 1), dtype=np.int16),
        charge_labels=None,
        charge_types=[U1Charge, U1Charge, U1Charge])

    q123 = q1 + q2 + q3

    nz_1 = np.nonzero(q123 == target)[0]
    i1 = fused_1 == target.charges[0, 0]
    i2 = fused_2 == target.charges[1, 0]
    i3 = fused_3 == target.charges[2, 0]
    #pylint: disable=no-member
    nz_2 = np.nonzero(np.logical_and.reduce([i1, i2, i3]))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()
  assert np.all(nz_1 == nz_2)


def test_U1Charge_multiple_fusion_with_flow():

  def run_test():
    D = 300
    B = 4
    O1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
    O2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
    O3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
    P1 = np.random.randint(0, B + 1, D).astype(np.int16)
    P2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    P3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    Q1 = np.random.randint(1, B + 1, D).astype(np.int8)
    Q2 = np.random.randint(0, B + 1, D).astype(np.int8)
    Q3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)

    charges_1 = [O1, -O2, O3]
    charges_2 = [P1, -P2, P3]
    charges_3 = [Q1, -Q2, Q3]

    fused_1 = fuse_ndarrays(charges_1)
    fused_2 = fuse_ndarrays(charges_2)
    fused_3 = fuse_ndarrays(charges_3)
    q1 = U1Charge(O1) @ U1Charge(P1) @ U1Charge(Q1)
    q2 = U1Charge(O2) @ U1Charge(P2) @ U1Charge(Q2)
    q3 = U1Charge(O3) @ U1Charge(P3) @ U1Charge(Q3)

    target = BaseCharge(
        charges=np.random.randint(-B, B, (3, 1), dtype=np.int16),
        charge_labels=None,
        charge_types=[U1Charge, U1Charge, U1Charge])
    q123 = q1 + q2 * True + q3
    nz_1 = np.nonzero(q123 == target)[0]
    i1 = fused_1 == target.charges[0, 0]
    i2 = fused_2 == target.charges[1, 0]
    i3 = fused_3 == target.charges[2, 0]
    #pylint: disable=no-member
    nz_2 = np.nonzero(np.logical_and.reduce([i1, i2, i3]))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()
  assert np.all(nz_1 == nz_2)


def test_U1Charge_fusion_with_flow():

  def run_test():
    D = 2000
    B = 6
    O1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
    O2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
    P1 = np.random.randint(0, B + 1, D).astype(np.int16)
    P2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
    Q1 = np.random.randint(1, B + 1, D).astype(np.int8)
    Q2 = np.random.randint(1, B + 1, D).astype(np.int8)

    charges_1 = [O1, -O2]
    charges_2 = [P1, -P2]
    charges_3 = [Q1, -Q2]

    fused_1 = fuse_ndarrays(charges_1)
    fused_2 = fuse_ndarrays(charges_2)
    fused_3 = fuse_ndarrays(charges_3)

    q1 = U1Charge(O1) @ U1Charge(P1) @ U1Charge(Q1)
    q2 = U1Charge(O2) @ U1Charge(P2) @ U1Charge(Q2)

    target = BaseCharge(
        charges=np.random.randint(-B, B, (3, 1), dtype=np.int16),
        charge_labels=None,
        charge_types=[U1Charge, U1Charge, U1Charge])
    q12 = q1 + q2 * True

    nz_1 = np.nonzero(q12 == target)[0]
    i1 = fused_1 == target.charges[0, 0]
    i2 = fused_2 == target.charges[1, 0]
    i3 = fused_3 == target.charges[2, 0]
    #pylint: disable=no-member
    nz_2 = np.nonzero(np.logical_and.reduce([i1, i2, i3]))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()
  assert np.all(nz_1 == nz_2)


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


def test_BaseCharge_eq():
  D = 3000
  B = 5
  c1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  c2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  q1 = U1Charge(c1)
  q2 = U1Charge(c2)
  Q = q1 @ q2
  target_charge = np.asarray([
      np.random.randint(-B // 2, B // 2 + 1, dtype=np.int16),
      np.random.randint(-B // 2 - 1, B // 2 + 2, dtype=np.int16)
  ])
  T = U1Charge(np.asarray([target_charge[0]])) @ U1Charge(
      np.asarray([target_charge[1]]))
  assert np.all(
      (np.squeeze(Q == T)
      ) == np.logical_and(c1 == target_charge[0], c2 == target_charge[1]))


def test_BaseCharge_unique():
  D = 3000
  B = 5
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  Q = BaseCharge(charges=q, charge_types=[U1Charge, U1Charge])
  expected = np.unique(
      q, return_index=True, return_inverse=True, return_counts=True, axis=1)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  assert np.all(actual[0].charges == expected[0])
  assert np.all(actual[1] == expected[1])
  assert np.all(actual[2] == expected[2])
  assert np.all(actual[3] == expected[3])


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


def test_intersect_3():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out, la, lb = intersect(a, b, return_indices=True)
  np.testing.assert_allclose([0, 4], out)
  np.testing.assert_allclose(la, [0, 4])
  np.testing.assert_allclose(lb, [0, 2])
