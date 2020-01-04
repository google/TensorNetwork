import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_tensor.charge import ChargeCollection, BaseCharge, U1Charge, Z2Charge
from tensornetwork.block_tensor.index import fuse_charges, fuse_degeneracies, fuse_charge_pair


def test_fuse_charge_pair():
  q1 = np.asarray([0, 1])
  q2 = np.asarray([2, 3, 4])
  fused_charges = fuse_charge_pair(q1, 1, q2, 1)
  assert np.all(fused_charges == np.asarray([2, 3, 4, 3, 4, 5]))
  fused_charges = fuse_charge_pair(q1, 1, q2, -1)
  assert np.all(fused_charges == np.asarray([-2, -3, -4, -1, -2, -3]))


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

  merged_charges = np.left_shift(charges[0].astype(np.int64),
                                 16) + charges[1].astype(np.int64)

  q1 = U1Charge(charges)
  assert np.all(q1.charges == merged_charges)


def test_U1Charge_dual():
  D = 100
  B = 6
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  merged_charges = np.left_shift(charges[0].astype(np.int64),
                                 16) + charges[1].astype(np.int64)

  q1 = U1Charge(charges)
  assert np.all(q1.dual_charges == -merged_charges)


def test_BaseCharge_raises():
  D = 100
  B = 6
  with pytest.raises(TypeError):
    q1 = BaseCharge([
        np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int64)
        for _ in range(2)
    ])
  with pytest.raises(ValueError):
    q1 = U1Charge([
        np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
        for _ in range(2)
    ],
                  shifts=[16, 0])


def test_U1Charge_fusion():

  def run_test():
    D = 2000
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

    nz_1 = np.nonzero(q12.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()

  assert np.all(nz_1 == nz_2)


def test_U1Charge_multiple_fusion():

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

    charges_1 = [O1, O2, O3]
    charges_2 = [P1, P2, P3]
    charges_3 = [Q1, Q2, Q3]

    fused_1 = fuse_charges(charges_1, [1, 1, 1])
    fused_2 = fuse_charges(charges_2, [1, 1, 1])
    fused_3 = fuse_charges(charges_3, [1, 1, 1])
    q1 = U1Charge([O1, P1, Q1])
    q2 = U1Charge([O2, P2, Q2])
    q3 = U1Charge([O3, P3, Q3])

    target = np.random.randint(-B // 2, B // 2 + 1, 3)
    q123 = q1 + q2 + q3

    nz_1 = np.nonzero(q123.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
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

    charges_1 = [O1, O2, O3]
    charges_2 = [P1, P2, P3]
    charges_3 = [Q1, Q2, Q3]

    fused_1 = fuse_charges(charges_1, [1, -1, 1])
    fused_2 = fuse_charges(charges_2, [1, -1, 1])
    fused_3 = fuse_charges(charges_3, [1, -1, 1])
    q1 = U1Charge([O1, P1, Q1])
    q2 = U1Charge([O2, P2, Q2])
    q3 = U1Charge([O3, P3, Q3])

    target = np.random.randint(-B // 2, B // 2 + 1, 3)
    q123 = q1 + (-1) * q2 + q3

    nz_1 = np.nonzero(q123.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
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

    charges_1 = [O1, O2]
    charges_2 = [P1, P2]
    charges_3 = [Q1, Q2]

    fused_1 = fuse_charges(charges_1, [1, -1])
    fused_2 = fuse_charges(charges_2, [1, -1])
    fused_3 = fuse_charges(charges_3, [1, -1])
    q1 = U1Charge([O1, P1, Q1])
    q2 = U1Charge([O2, P2, Q2])

    target = np.random.randint(-B // 2, B // 2 + 1, 3)
    q12 = q1 + (-1) * q2

    nz_1 = np.nonzero(q12.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()
  assert np.all(nz_1 == nz_2)


def test_U1Charge_sub():

  def run_test():
    D = 2000
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

    fused_1 = fuse_charges(charges_1, [1, -1])
    fused_2 = fuse_charges(charges_2, [1, -1])
    fused_3 = fuse_charges(charges_3, [1, -1])
    q1 = U1Charge([O1, P1, Q1])
    q2 = U1Charge([O2, P2, Q2])

    target = np.random.randint(-B // 2, B // 2 + 1, 3)
    q12 = q1 - q2

    nz_1 = np.nonzero(q12.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
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

  q1 = U1Charge([C1])
  q2 = U1Charge([C2])
  q3 = U1Charge([C3])

  Q = q1 @ q2 @ q3
  Q_ = U1Charge([C1, C2, C3])
  assert np.all(Q.charges == Q_.charges)
  #assert Q.offsets == Q_.offsets
  assert np.all(Q.shifts == Q_.shifts)


def test_Z2Charge_fusion():

  def fuse_z2_charges(c1, c2):
    return np.reshape(
        np.bitwise_xor(c1[:, None], c2[None, :]),
        len(c1) * len(c2))

  def run_test():
    D = 1000
    O1 = np.random.randint(0, 2, D).astype(np.int8)
    O2 = np.random.randint(0, 2, D).astype(np.int8)
    P1 = np.random.randint(0, 2, D).astype(np.int8)
    P2 = np.random.randint(0, 2, D).astype(np.int8)
    Q1 = np.random.randint(0, 2, D).astype(np.int8)
    Q2 = np.random.randint(0, 2, D).astype(np.int8)

    charges_1 = [O1, O2]
    charges_2 = [P1, P2]
    charges_3 = [Q1, Q2]

    fused_1 = fuse_z2_charges(*charges_1)
    fused_2 = fuse_z2_charges(*charges_2)
    fused_3 = fuse_z2_charges(*charges_3)

    q1 = Z2Charge([O1, P1, Q1])
    q2 = Z2Charge([O2, P2, Q2])

    target = np.random.randint(0, 2, 3)
    q12 = q1 + q2

    nz_1 = np.nonzero(q12.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()
  assert np.all(nz_1 == nz_2)


def test_Z2Charge_sub():

  def fuse_z2_charges(c1, c2):
    return np.reshape(
        np.bitwise_xor(c1[:, None], c2[None, :]),
        len(c1) * len(c2))

  def run_test():
    D = 1000
    O1 = np.random.randint(0, 2, D).astype(np.int8)
    O2 = np.random.randint(0, 2, D).astype(np.int8)
    P1 = np.random.randint(0, 2, D).astype(np.int8)
    P2 = np.random.randint(0, 2, D).astype(np.int8)
    Q1 = np.random.randint(0, 2, D).astype(np.int8)
    Q2 = np.random.randint(0, 2, D).astype(np.int8)

    charges_1 = [O1, O2]
    charges_2 = [P1, P2]
    charges_3 = [Q1, Q2]

    fused_1 = fuse_z2_charges(*charges_1)
    fused_2 = fuse_z2_charges(*charges_2)
    fused_3 = fuse_z2_charges(*charges_3)

    q1 = Z2Charge([O1, P1, Q1])
    q2 = Z2Charge([O2, P2, Q2])

    target = np.random.randint(0, 2, 3)
    q12 = q1 - q2

    nz_1 = np.nonzero(q12.equals(target))[0]
    i1 = fused_1 == target[0]
    i2 = fused_2 == target[1]
    i3 = fused_3 == target[2]
    nz_2 = np.nonzero(np.logical_and(np.logical_and(i1, i2), i3))[0]
    return nz_1, nz_2

  nz_1, nz_2 = run_test()
  while len(nz_1) == 0:
    nz_1, nz_2 = run_test()
  assert np.all(nz_1 == nz_2)


def test_Z2Charge_matmul():
  D = 1000
  C1 = np.random.randint(0, 2, D).astype(np.int8)
  C2 = np.random.randint(0, 2, D).astype(np.int8)
  C3 = np.random.randint(0, 2, D).astype(np.int8)

  q1 = Z2Charge([C1])
  q2 = Z2Charge([C2])
  q3 = Z2Charge([C3])

  Q = q1 @ q2 @ q3
  Q_ = Z2Charge([C1, C2, C3])
  assert np.all(Q.charges == Q_.charges)
  assert np.all(Q.shifts == Q_.shifts)


def test_Charge_U1_add():
  q1 = ChargeCollection(
      [U1Charge([np.asarray([0, 1])]),
       U1Charge([np.asarray([-2, 3])])])
  q2 = ChargeCollection(
      [U1Charge([np.asarray([2, 3])]),
       U1Charge([np.asarray([-1, 4])])])
  expected = [np.asarray([2, 3, 3, 4]), np.asarray([-3, 2, 2, 7])]
  q12 = q1 + q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_U1_sub():
  q1 = ChargeCollection(
      [U1Charge([np.asarray([0, 1])]),
       U1Charge([np.asarray([-2, 3])])])
  q2 = ChargeCollection(
      [U1Charge([np.asarray([2, 3])]),
       U1Charge([np.asarray([-1, 4])])])
  expected = [np.asarray([-2, -3, -1, -2]), np.asarray([-1, -6, 4, -1])]
  q12 = q1 - q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_Z2_add():
  q1 = ChargeCollection([
      Z2Charge([np.asarray([0, 1]).astype(np.int8)]),
      Z2Charge([np.asarray([1, 0]).astype(np.int8)])
  ])
  q2 = ChargeCollection([
      Z2Charge([np.asarray([0, 0]).astype(np.int8)]),
      Z2Charge([np.asarray([1, 1]).astype(np.int8)])
  ])
  expected = [np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 1, 1])]
  q12 = q1 + q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_Z2_sub():
  q1 = ChargeCollection([
      Z2Charge([np.asarray([0, 1]).astype(np.int8)]),
      Z2Charge([np.asarray([1, 0]).astype(np.int8)])
  ])
  q2 = ChargeCollection([
      Z2Charge([np.asarray([0, 0]).astype(np.int8)]),
      Z2Charge([np.asarray([1, 1]).astype(np.int8)])
  ])
  expected = [np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 1, 1])]
  q12 = q1 - q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_Z2_U1_add():
  q1 = ChargeCollection([
      Z2Charge([np.asarray([0, 1]).astype(np.int8)]),
      U1Charge([np.asarray([-2, 3]).astype(np.int8)])
  ])
  q2 = ChargeCollection([
      Z2Charge([np.asarray([0, 0]).astype(np.int8)]),
      U1Charge([np.asarray([-1, 4]).astype(np.int8)])
  ])
  expected = [np.asarray([0, 0, 1, 1]), np.asarray([-3, 2, 2, 7])]

  q12 = q1 + q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_add_Z2_U1_raises():
  q1 = ChargeCollection([
      Z2Charge([np.asarray([0, 1]).astype(np.int8)]),
      Z2Charge([np.asarray([-2, 3]).astype(np.int8)])
  ])
  q2 = ChargeCollection(
      [U1Charge([np.asarray([0, 0])]),
       U1Charge([np.asarray([-1, 4])])])
  expected = [np.asarray([0, 0, 1, 1]), np.asarray([-3, 2, 2, 7])]
  with pytest.raises(TypeError):
    q12 = q1 + q2


def test_Charge_sub_Z2_U1_raises():
  q1 = ChargeCollection([
      Z2Charge([np.asarray([0, 1]).astype(np.int8)]),
      Z2Charge([np.asarray([-2, 3]).astype(np.int8)])
  ])
  q2 = ChargeCollection(
      [U1Charge([np.asarray([0, 0])]),
       U1Charge([np.asarray([-1, 4])])])
  expected = [np.asarray([0, 0, 1, 1]), np.asarray([-3, 2, 2, 7])]
  with pytest.raises(TypeError):
    q12 = q1 - q2


def test_BaseCharge_eq():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  Q = BaseCharge(charges=[q1, q2])
  target_charge = np.asarray([
      np.random.randint(-B // 2, B // 2 + 1),
      np.random.randint(-B // 2 - 1, B // 2 + 2)
  ])
  assert np.all(
      (Q == np.left_shift(target_charge[0], 16) + target_charge[1]
      ) == np.logical_and(q1 == target_charge[0], q2 == target_charge[1]))


def test_BaseCharge_equals():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  Q = BaseCharge(charges=[q1, q2])
  target_charge = np.asarray([
      np.random.randint(-B // 2, B // 2 + 1),
      np.random.randint(-B // 2 - 1, B // 2 + 2)
  ])
  assert np.all(
      (Q.equals(target_charge)
      ) == np.logical_and(q1 == target_charge[0], q2 == target_charge[1]))


def test_BaseCharge_unique():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  Q = BaseCharge(charges=[q1, q2])
  expected = np.unique(
      Q.charges,
      return_index=True,
      return_inverse=True,
      return_counts=True,
      axis=0)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  assert np.all(actual[0].charges == expected[0])
  assert np.all(actual[1] == expected[1])
  assert np.all(actual[2] == expected[2])
  assert np.all(actual[3] == expected[3])


def test_Charge_U1_U1_equals():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  p1 = np.random.randint(-B // 2 - 2, B // 2 + 3, D).astype(np.int16)
  Q = ChargeCollection(charges=[U1Charge([q1, q2]), U1Charge(p1)])
  target_q = [
      np.random.randint(-B // 2, B // 2 + 1),
      np.random.randint(-B // 2 - 1, B // 2 + 2)
  ]
  target_p = [np.random.randint(-B // 2 - 2, B // 2 + 3)]
  target_charge = [target_q, target_p]
  assert np.all((Q.equals(target_charge)) == np.logical_and.reduce(
      [q1 == target_q[0], q2 == target_q[1], p1 == target_p[0]]))


def test_Charge_U1_U1_eq():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  p1 = np.random.randint(-B // 2 - 2, B // 2 + 3, D).astype(np.int16)
  Q = ChargeCollection(charges=[U1Charge([q1, q2]), U1Charge(p1)])
  target_q = [
      np.random.randint(-B // 2, B // 2 + 1),
      np.random.randint(-B // 2 - 1, B // 2 + 2)
  ]
  target_q_shifted = np.left_shift(target_q[0], 16) + target_q[1]
  target_p = [np.random.randint(-B // 2 - 2, B // 2 + 3)]
  target_charge = [target_q_shifted, target_p]
  assert np.all((Q == target_charge) == np.logical_and.reduce(
      [q1 == target_q[0], q2 == target_q[1], p1 == target_p[0]]))


def test_Charge_Z2_Z2_equals():
  D = 3000
  q1 = np.random.randint(0, 2, D).astype(np.int8)
  q2 = np.random.randint(0, 2, D).astype(np.int8)
  p1 = np.random.randint(0, 2, D).astype(np.int8)
  Q = ChargeCollection(charges=[Z2Charge([q1, q2]), Z2Charge(p1)])
  target_q = [np.random.randint(0, 2), np.random.randint(0, 2)]
  target_p = [np.random.randint(0, 2)]
  target_charge = [target_q, target_p]
  assert np.all((Q.equals(target_charge)) == np.logical_and.reduce(
      [q1 == target_q[0], q2 == target_q[1], p1 == target_p[0]]))


def test_Charge_Z2_Z2_eq():
  D = 3000
  q1 = np.random.randint(0, 2, D).astype(np.int8)
  q2 = np.random.randint(0, 2, D).astype(np.int8)
  p1 = np.random.randint(0, 2, D).astype(np.int8)
  Q = ChargeCollection(charges=[Z2Charge([q1, q2]), Z2Charge(p1)])
  target_q = [np.random.randint(0, 2), np.random.randint(0, 2)]
  target_q_shifted = np.left_shift(target_q[0], 8) + target_q[1]
  target_p = [np.random.randint(0, 2)]
  target_charge = [target_q_shifted, target_p]
  assert np.all((Q == target_charge) == np.logical_and.reduce(
      [q1 == target_q[0], q2 == target_q[1], p1 == target_p[0]]))


def test_Charge_U1_Z2_equals():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  p1 = np.random.randint(0, 2, D).astype(np.int8)
  Q = ChargeCollection(charges=[U1Charge([q1, q2]), Z2Charge(p1)])
  target_q = [
      np.random.randint(-B // 2, B // 2 + 1),
      np.random.randint(-B // 2 - 1, B // 2 + 2)
  ]
  target_p = [np.random.randint(0, 2)]
  target_charge = [target_q, target_p]
  assert np.all((Q.equals(target_charge)) == np.logical_and.reduce(
      [q1 == target_q[0], q2 == target_q[1], p1 == target_p[0]]))


def test_Charge_U1_Z2_eq():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  p1 = np.random.randint(0, 2, D).astype(np.int8)
  Q = ChargeCollection(charges=[U1Charge([q1, q2]), Z2Charge(p1)])
  target_q = [
      np.random.randint(-B // 2, B // 2 + 1),
      np.random.randint(-B // 2 - 1, B // 2 + 2)
  ]
  target_q_shifted = np.left_shift(target_q[0], 16) + target_q[1]
  target_p = [np.random.randint(0, 2)]
  target_charge = [target_q_shifted, target_p]
  assert np.all((Q == target_charge) == np.logical_and.reduce(
      [q1 == target_q[0], q2 == target_q[1], p1 == target_p[0]]))


def test_Charge_U1_U1_unique():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  p1 = np.random.randint(-B // 2 - 2, B // 2 + 3, D).astype(np.int16)
  Q = ChargeCollection(charges=[U1Charge([q1, q2]), U1Charge(p1)])
  expected = np.unique(
      np.stack([Q.charges[0].charges, Q.charges[1].charges], axis=1),
      return_index=True,
      return_inverse=True,
      return_counts=True,
      axis=0)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  assert np.all(actual[0].charges[0].charges == expected[0][:, 0])
  assert np.all(actual[0].charges[1].charges == expected[0][:, 1])
  assert np.all(actual[1] == expected[1])
  assert np.all(actual[2] == expected[2])
  assert np.all(actual[3] == expected[3])


def test_Charge_Z2_Z2_unique():
  D = 3000
  B = 5
  q1 = np.random.randint(0, 2, D).astype(np.int8)
  q2 = np.random.randint(0, 2, D).astype(np.int8)
  p1 = np.random.randint(0, 2, D).astype(np.int8)
  Q = ChargeCollection(charges=[Z2Charge([q1, q2]), Z2Charge(p1)])
  expected = np.unique(
      np.stack([Q.charges[0].charges, Q.charges[1].charges], axis=1),
      return_index=True,
      return_inverse=True,
      return_counts=True,
      axis=0)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  assert np.all(actual[0].charges[0].charges == expected[0][:, 0])
  assert np.all(actual[0].charges[1].charges == expected[0][:, 1])
  assert np.all(actual[1] == expected[1])
  assert np.all(actual[2] == expected[2])
  assert np.all(actual[3] == expected[3])


def test_Charge_U1_Z2_unique():
  D = 3000
  B = 5
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
  q2 = np.random.randint(-B // 2 - 1, B // 2 + 2, D).astype(np.int16)
  p1 = np.random.randint(0, 2, D).astype(np.int8)
  Q = ChargeCollection(charges=[U1Charge([q1, q2]), Z2Charge(p1)])
  expected = np.unique(
      np.stack([Q.charges[0].charges, Q.charges[1].charges], axis=1),
      return_index=True,
      return_inverse=True,
      return_counts=True,
      axis=0)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  assert np.all(actual[0].charges[0].charges == expected[0][:, 0])
  assert np.all(actual[0].charges[1].charges == expected[0][:, 1])
  assert np.all(actual[1] == expected[1])
  assert np.all(actual[2] == expected[2])
  assert np.all(actual[3] == expected[3])
