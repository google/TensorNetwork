import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_tensor.charge import Charge, BaseCharge, U1Charge, Z2Charge
from tensornetwork.block_tensor.index import fuse_charges, fuse_degeneracies, fuse_charge_pair


def test_fuse_charge_pair():
  q1 = np.asarray([0, 1])
  q2 = np.asarray([2, 3, 4])
  fused_charges = fuse_charge_pair(q1, 1, q2, 1)
  assert np.all(fused_charges == np.asarray([2, 3, 4, 3, 4, 5]))
  fused_charges = fuse_charge_pair(q1, 1, q2, -1)
  assert np.all(fused_charges == np.asarray([-2, -3, -4, -1, -2, -3]))


def test_Charge_add():
  q1 = Charge([U1Charge([np.asarray([0, 1])]), U1Charge([np.asarray([-2, 3])])])
  q2 = Charge([U1Charge([np.asarray([2, 3])]), U1Charge([np.asarray([-1, 4])])])
  expected = [np.asarray([2, 3, 3, 4]), np.asarray([-3, 2, 2, 7])]
  q12 = q1 + q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_get_charges():
  q = Charge(
      [U1Charge([np.asarray([-1, 0, 1])]),
       U1Charge([np.asarray([-2, 0, 3])])])
  expected = [np.asarray([-1, 0, 1]), np.asarray([-2, 0, 3])]
  actual = q.get_charges(dual=False)
  for n in range(len(actual)):
    np.testing.assert_allclose(expected[n], actual[n])

  expected = [np.asarray([1, 0, -1]), np.asarray([2, 0, -3])]
  actual = q.get_charges(dual=True)
  for n in range(len(actual)):
    np.testing.assert_allclose(expected[n], actual[n])


def test_fuse_charges():
  q1 = np.asarray([0, 1])
  q2 = np.asarray([2, 3, 4])
  fused_charges = fuse_charges([q1, q2], flows=[1, 1])
  assert np.all(fused_charges == np.asarray([2, 3, 4, 3, 4, 5]))
  fused_charges = fuse_charges([q1, q2], flows=[1, -1])
  assert np.all(fused_charges == np.asarray([-2, -3, -4, -1, -2, -3]))


def test_fuse_degeneracies():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused_degeneracies = fuse_degeneracies(d1, d2)
  np.testing.assert_allclose(fused_degeneracies, np.kron(d1, d2))


def test_U1Charge_charges():
  D = 100
  B = 6
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]

  offsets = [np.min([0, np.min(c)]) for c in charges]
  pos_charges = [
      charges[n].astype(np.int32) - offsets[n].astype(np.int32)
      for n in range(2)
  ]
  merged_charges = np.left_shift(pos_charges[0], 16) + pos_charges[1]
  merged_offsets = np.left_shift(offsets[0], 16) + offsets[1]

  q1 = U1Charge(charges)
  assert np.all(q1.charges == merged_charges + merged_offsets)


def test_U1Charge_dual():
  D = 100
  B = 6
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]

  offsets = [np.min([0, np.min(c)]) for c in charges]
  pos_charges = [
      charges[n].astype(np.int32) - offsets[n].astype(np.int32)
      for n in range(2)
  ]
  merged_charges = np.left_shift(pos_charges[0], 16) + pos_charges[1]
  merged_offsets = np.left_shift(offsets[0], 16) + offsets[1]

  q1 = U1Charge(charges)
  assert np.all(q1.dual_charges == -(merged_charges + merged_offsets))


def test_U1Charge_get_charges():
  D = 100
  B = 6
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  q1 = U1Charge(charges)
  assert np.all(q1.get_charges(False) == q1.charges)
  assert np.all(q1.get_charges(True) == q1.dual_charges)


def test_U1Charge_raises():
  D = 100
  B = 6
  with pytest.raises(TypeError):
    q1 = U1Charge([
        np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int64)
        for _ in range(2)
    ])
  with pytest.raises(ValueError):
    q1 = U1Charge([
        np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int64)
        for _ in range(2)
    ],
                  offsets=[-5, -6])
  with pytest.raises(ValueError):
    q1 = U1Charge([
        np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int64)
        for _ in range(2)
    ],
                  shifts=[16, 0])
  with pytest.raises(ValueError):
    q1 = U1Charge([
        np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int64)
        for _ in range(2)
    ],
                  offsets=[-5, -6],
                  shifts=[16, 0])


def test_U1Charge_fusion():
  D = 1000
  B = 6
  O1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
  O2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int8)
  P1 = np.random.randint(0, B + 1, D).astype(np.int16)
  P2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  Q1 = np.random.randint(1, B + 1, D).astype(np.int8)
  Q2 = np.random.randint(1, B + 1, D).astype(np.int8)

  charges_1 = [O1, O2]
  charges_2 = [P1, P2]
  charges_3 = [Q1, Q2]

  fused_1 = fuse_charges(charges_1, [1, 1])
  fused_2 = fuse_charges(charges_2, [1, 1])
  fused_3 = fuse_charges(charges_3, [1, 1])
  q1 = U1Charge([O1, P1, Q1])
  q2 = U1Charge([O2, P2, Q2])

  target = np.random.randint(-B // 2, B // 2 + 1, 3)
  q12 = q1 + q2

  nz_1 = q12.nonzero(target)
  i1 = fused_1 == target[0]
  i2 = fused_2 == target[1]
  i3 = fused_3 == target[2]
  nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
  assert np.all(nz_1 == nz_2)


def test_U1Charge_matmul():
  D = 1000
  B = 5
  C1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  C2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  C3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)

  q1 = U1Charge([C1])
  q2 = U1Charge([C2])
  q3 = U1Charge([C3])

  Q = q1 @ q2 @ q3
  Q_ = U1Charge([C1, C2, C3])
  assert np.all(Q.charges == Q_.charges)
  assert np.all(Q._charges == Q_._charges)
  assert Q.offsets == Q_.offsets
  assert np.all(Q.shifts == Q_.shifts)


def test_Z2Charge_fusion():
  D = 1000
  B = 6
  O1 = np.random.randint(0, 2, D).astype(np.int8)
  O2 = np.random.randint(0, 2, D).astype(np.int8)
  P1 = np.random.randint(0, 2, D).astype(np.int8)
  P2 = np.random.randint(0, 2, D).astype(np.int8)
  Q1 = np.random.randint(0, 2, D).astype(np.int8)
  Q2 = np.random.randint(0, 2, D).astype(np.int8)

  charges_1 = [O1, O2]
  charges_2 = [P1, P2]
  charges_3 = [Q1, Q2]

  def fuse_z2_charges(c1, c2):
    return np.reshape(
        np.bitwise_xor(c1[:, None], c2[None, :]),
        len(c1) * len(c2))

  fused_1 = fuse_z2_charges(*charges_1)
  fused_2 = fuse_z2_charges(*charges_2)
  fused_3 = fuse_z2_charges(*charges_3)

  q1 = Z2Charge([O1, P1, Q1])
  q2 = Z2Charge([O2, P2, Q2])

  target = np.random.randint(0, 2, 3)
  q12 = q1 + q2

  nz_1 = q12.nonzero(target)
  i1 = fused_1 == target[0]
  i2 = fused_2 == target[1]
  i3 = fused_3 == target[2]
  nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
  assert np.all(nz_1 == nz_2)


def test_U1Charge_matmul():
  D = 1000
  B = 5
  C1 = np.random.randint(0, 2, D).astype(np.int8)
  C2 = np.random.randint(0, 2, D).astype(np.int8)
  C3 = np.random.randint(0, 2, D).astype(np.int8)

  q1 = Z2Charge([C1])
  q2 = Z2Charge([C2])
  q3 = Z2Charge([C3])

  Q = q1 @ q2 @ q3
  Q_ = Z2Charge([C1, C2, C3])
  assert np.all(Q.charges == Q_.charges)
  assert np.all(Q._charges == Q_._charges)
  assert np.all(Q.shifts == Q_.shifts)
