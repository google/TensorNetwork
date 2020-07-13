import numpy as np
from tensornetwork.block_sparse.charge import (collapse, expand, BaseCharge,
                                               U1Charge, Z2Charge, ZNCharge,
                                               intersect, fuse_ndarrays,
                                               fuse_degeneracies,
                                               fuse_charges)
import pytest

# pylint: disable=unbalanced-tuple-unpacking
def test_charge_collapse_expand_1():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  [a1f, b1f] = expand(a1b1, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  assert a1b1.dtype == np.int32
  a1f, b1f = expand(a1b1, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  assert a1b1.dtype == np.int32
  a1f, b1f = expand(a1b1, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = expand(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = expand(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = expand(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = expand(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)


def test_charge_collapse_expand_2():
  np.random.seed(10)
  a = np.random.randint(-10, 10, 10000).astype(np.int8)
  b = np.random.randint(-10, 10, 10000).astype(np.int16)
  c = np.random.randint(-10, 10, 10000).astype(np.int8)

  r1 = np.left_shift(a.astype(np.int32), 24) + np.left_shift(
      b.astype(np.int32), 8) + c.astype(np.int32)
  r2 = np.left_shift(a.astype(np.int32), 24) + (
      np.left_shift(b.astype(np.int32), 8) + c.astype(np.int32))
  np.testing.assert_allclose(r1, r2)

  cf = np.bitwise_and(r1, 2**8 - 1).astype(np.int8)
  np.testing.assert_allclose(cf, c)
  r1 = np.right_shift(r1 - cf.astype(np.int32), 8)
  bf = np.bitwise_and(r1, 2**16 - 1).astype(np.int16)
  np.testing.assert_allclose(bf, b)
  af = np.right_shift(r1 - bf.astype(np.int32), 16).astype(np.int8)
  np.testing.assert_allclose(af, a)

  bc = collapse([b, c], [[b.dtype], [c.dtype]])
  abc = collapse([a, bc], [[a.dtype], [b.dtype, c.dtype]])
  abc2 = collapse([a, b, c], [[a.dtype], [b.dtype], [c.dtype]])
  np.testing.assert_allclose(abc, r2)
  np.testing.assert_allclose(abc2, r2)


def test_charge_collapse_expand_3():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 10000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 10000).astype(np.int8)
  a2 = np.random.randint(-10, 10, 10000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 10000).astype(np.int8)

  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a2b2 = collapse([a2, b2], [[a2.dtype], [b2.dtype]])

  a1b1a2b2 = collapse([a1b1, a2b2],
                      [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1b1_, a2b2_ = expand(a1b1a2b2, [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1f, b1f = expand(a1b1_, [[a1.dtype], [b1.dtype]])
  a2f, b2f = expand(a2b2_, [[a2.dtype], [b2.dtype]])

  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(a2f, a2)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(b2f, b2)


def test_charge_collapse_expand_4():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 10000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 10000).astype(np.int16)
  a2 = np.random.randint(-10, 10, 10000).astype(np.int8)

  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a1b1a2 = collapse([a1b1, a2], [[a1.dtype, b1.dtype], [a2.dtype]])
  a1b1_, a2f = expand(a1b1a2, [[a1.dtype, b1.dtype], [a2.dtype]])
  a1f, b1f = expand(a1b1_, [[a1.dtype], [b1.dtype]])

  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(a2f, a2)
  np.testing.assert_allclose(b1f, b1)


def test_charge_collapse_expand_5():
  np.random.seed(10)
  a = np.random.randint(-10, 10, 10000).astype(np.int8)
  b = np.random.randint(-10, 10, 10000).astype(np.int16)
  c = np.random.randint(-10, 10, 10000).astype(np.int8)

  bc = collapse([b, c], [[b.dtype], [c.dtype]])
  abc = collapse([a, bc], [[a.dtype], [b.dtype, c.dtype]])
  abc2 = collapse([a, b, c], [[a.dtype], [b.dtype], [c.dtype]])
  r1 = np.left_shift(a.astype(np.int32), 24) + np.left_shift(
      b.astype(np.int32), 8) + c.astype(np.int32)
  np.testing.assert_allclose(abc, r1)
  np.testing.assert_allclose(abc2, abc)

  af, bf, cf = expand(abc, [[a.dtype], [b.dtype], [c.dtype]])
  np.testing.assert_allclose(af, a)
  np.testing.assert_allclose(bf, b)
  np.testing.assert_allclose(cf, c)
  af, bcf = expand(abc, [[a.dtype], [b.dtype, c.dtype]])
  bf, cf = expand(bcf, [[b.dtype], [c.dtype]])
  np.testing.assert_allclose(af, a)
  np.testing.assert_allclose(bf, b)
  np.testing.assert_allclose(cf, c)


def test_charge_collapse_expand_6():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 1000000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 1000000).astype(np.int16)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a2 = np.random.randint(-10, 10, 1000000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 1000000).astype(np.int16)
  a2b2 = collapse([a2, b2], [[a2.dtype], [b2.dtype]])
  a1b1a2b2 = collapse([a1b1, a2b2],
                      [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1b1_, a2b2_ = expand(a1b1a2b2, [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1f, b1f = expand(a1b1_, [[a1.dtype], [b1.dtype]])
  a2f, b2f = expand(a2b2_, [[a2.dtype], [b2.dtype]])

  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(a2f, a2)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(b2f, b2)


def test_charge_collapse_expand_7():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  a2 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 100000).astype(np.int16)

  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a2b2 = collapse([a2, b2], [[a2.dtype], [b2.dtype]])
  res = a1b1 + a2b2
  resa, resb = expand(res, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(resa, a1 + a2)
  np.testing.assert_allclose(resb, b1 + b2)


def test_charge_collapse_expand_8():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a2 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 100000).astype(np.int16)
  c2 = np.random.randint(-10, 10, 100000).astype(np.int8)

  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  a2b2c2 = collapse([a2, b2, c2], [[a2.dtype], [b2.dtype], [c2.dtype]])
  res = a1b1c1 + a2b2c2
  resa, _ = expand(res, [[a1.dtype], [b1.dtype, c1.dtype]])
  resb, resc = expand(res, [[b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(resa, a1 + a2)
  np.testing.assert_allclose(resb, b1 + b2)
  np.testing.assert_allclose(resc, c1 + c2)
  resa2, resb2, resc2 = expand(res, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(resa2, a1 + a2)
  np.testing.assert_allclose(resb2, b1 + b2)
  np.testing.assert_allclose(resc2, c1 + c2)


def test_BaseCharge_charges():
  D = 100
  B = 6
  np.random.seed(10)
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  q1 = BaseCharge(charges)
  q1.expand_charge_types()
  np.testing.assert_allclose(
      np.stack(q1.charges, axis=0), np.stack(charges, axis=0))


def test_BaseCharge_generic():
  D = 300
  B = 5
  np.random.seed(10)
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  Q = BaseCharge(charges)
  assert Q.dim == 300
  assert Q.num_symmetries == 2


def test_BaseCharge_len():
  D = 300
  B = 5
  np.random.seed(10)
  q = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  Q = BaseCharge(charges=q)
  assert len(Q) == 300


def test_BaseCharge_copy():
  D = 300
  B = 5
  np.random.seed(10)
  q = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  Q = BaseCharge(charges=q)
  Qcopy = Q.copy()
  for n, c in enumerate(Q.charges):
    assert c is not Qcopy.charges[n]
  np.testing.assert_allclose(np.stack(Q.charges), Qcopy.charges)


def test_BaseCharge_unique():
  D = 3000
  B = 5
  np.random.seed(10)
  q = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
  Q = BaseCharge(charges=q, charge_types=[[U1Charge], [U1Charge]])
  expected = np.unique(
      np.stack(q, axis=1),
      return_index=True,
      return_inverse=True,
      return_counts=True,
      axis=0)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  actual[0].expand_charge_types()
  np.testing.assert_allclose(np.stack(actual[0].charges, axis=1), expected[0])
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


def test_intersect_4():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out, la, lb = intersect(a, b, return_indices=True)
  np.testing.assert_allclose([0, 4], out)
  np.testing.assert_allclose(la, [0, 4])
  np.testing.assert_allclose(lb, [0, 2])


def test_intersect_raises():
  np.random.seed(10)
  a = np.random.randint(0, 10, (4, 5))
  b = np.random.randint(0, 10, (4, 6))
  with pytest.raises(ValueError):
    intersect(a, b, axis=0)
  c = np.random.randint(0, 10, (3, 7))
  with pytest.raises(ValueError):
    intersect(a, c, axis=1)
  with pytest.raises(NotImplementedError):
    intersect(a, c, axis=2)
  d = np.random.randint(0, 10, (3, 7, 3))
  e = np.random.randint(0, 10, (3, 7, 3))
  with pytest.raises(NotImplementedError):
    intersect(d, e, axis=1)


def test_fuse_ndarrays():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused = fuse_ndarrays([d1, d2])
  np.testing.assert_allclose(fused, [2, 3, 4, 3, 4, 5])


def test_fuse_degeneracies():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused_degeneracies = fuse_degeneracies(d1, d2)
  np.testing.assert_allclose(fused_degeneracies, np.kron(d1, d2))


@pytest.mark.parametrize('chargetype, B0, B1', [(U1Charge, -5, 5),
                                                (Z2Charge, 0, 1),
                                                (ZNCharge(3), 0, 2),
                                                (ZNCharge(6), 0, 5)])
def test_Charge_charges(chargetype, B0, B1):
  D = 100
  np.random.seed(10)
  charges = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  q1 = chargetype(charges)
  np.testing.assert_allclose(q1.charges[0], charges)


@pytest.mark.parametrize('chargetype, B0, B1,sign', [(U1Charge, -5, 5, -1),
                                                     (Z2Charge, 0, 1, 1)])
def test_Charge_dual(chargetype, B0, B1, sign):
  D = 100
  np.random.seed(10)
  charges = np.random.randint(B0, B1 + 1, D).astype(np.int16)

  q1 = chargetype(charges)
  assert np.all(q1.dual(True).charges[0] == sign * charges)


@pytest.mark.parametrize('n', list(range(2, 12)))
def test_Charge_dual_zncharges(n):
  chargetype = ZNCharge(n)
  D = 100
  np.random.seed(10)
  charges = np.random.randint(0, n, D).astype(np.int16)
  q1 = chargetype(charges)
  assert np.all(q1.dual(True).charges[0] == (n - charges) % n)


def test_Z2Charge_raises():
  np.random.seed(10)
  charges = np.array([-1, 0, 1, 2])
  with pytest.raises(ValueError):
    Z2Charge(charges)


def get_charges(B0, B1, D, num_legs):
  return [
      np.random.randint(B0, B1 + 1, D).astype(np.int16) for _ in range(num_legs)
  ]


def fuse_many_charges(num_legs, num_charge_types, seed, D, B, use_flows=False):
  np.random.seed(seed)
  if use_flows:
    flows = np.random.choice([True, False], num_legs, replace=True)
  else:
    flows = np.asarray([False] * num_legs)
  np_flows = np.ones(num_legs, dtype=np.int16)
  np_flows[flows] = -1
  charges = [
      get_charges(-B // 2, B // 2, D, num_charge_types) for _ in range(num_legs)
  ]
  fused = [
      fuse_ndarrays([charges[n][m] * np_flows[n]
                     for n in range(num_legs)])
      for m in range(num_charge_types)
  ]
  final_charges = [U1Charge(charges[n][0]) for n in range(num_legs)]
  for n in range(num_legs):
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
  for n in range(1, num_legs):
    final = final + final_charges[n] * flows[n]

  nz_1 = np.nonzero(final == target)[0]
  masks = [fused[m] == np_target_charges[m] for m in range(num_charge_types)]

  #pylint: disable=no-member
  nz_2 = np.nonzero(np.logical_and.reduce(masks))[0]
  return nz_1, nz_2


@pytest.mark.parametrize('use_flows', [True, False])
@pytest.mark.parametrize('num_legs, num_charge_types, D, B', [(2, 1, 1000, 6),
                                                              (2, 2, 1000, 6),
                                                              (3, 1, 100, 6),
                                                              (3, 2, 100, 6),
                                                              (3, 3, 100, 6)])
def test_U1Charge_fusion(num_legs, num_charge_types, D, B, use_flows):
  nz_1, nz_2 = fuse_many_charges(
      num_legs=num_legs,
      num_charge_types=num_charge_types,
      seed=20,
      D=D,
      B=B,
      use_flows=use_flows)
  assert len(nz_1) > 0
  assert len(nz_2) > 0
  assert np.all(nz_1 == nz_2)


def test_BaseCharge_intersect():
  q1 = [
      np.array([0, 1, 2, 0, 6], dtype=np.int16),
      np.array([2, 3, 4, -1, 4], dtype=np.int16)
  ]
  q2 = [
      np.array([0, -2, 6], dtype=np.int16),
      np.array([2, 3, 4], dtype=np.int16)
  ]
  Q1 = BaseCharge(charges=q1)
  Q2 = BaseCharge(charges=q2)
  res = Q1.intersect(Q2)
  res.expand_charge_types()
  np.testing.assert_allclose(
      np.stack(res.charges, axis=0), np.asarray([[0, 6], [2, 4]]))


def test_BaseCharge_intersect_return_indices():
  q1 = [
      np.array([0, 1, 2, 0, 6], dtype=np.int16),
      np.array([2, 3, 4, -1, 4], dtype=np.int16)
  ]
  q2 = [
      np.array([-2, 0, 6], dtype=np.int16),
      np.array([3, 2, 4], dtype=np.int16)
  ]

  Q1 = BaseCharge(charges=q1)
  Q2 = BaseCharge(charges=q2)
  res, i1, i2 = Q1.intersect(Q2, return_indices=True)
  #res, i1, i2 = intersect(q1, q2, axis=1, return_indices=True)
  res.expand_charge_types()
  np.testing.assert_allclose(
      np.stack(res.charges, axis=0), np.asarray([[0, 6], [2, 4]]))
  np.testing.assert_allclose(i1, [0, 4])
  np.testing.assert_allclose(i2, [1, 2])


@pytest.mark.parametrize('chargetype, B0, B1, dtype',
                         [(U1Charge, -5, 5, np.int16),
                          (Z2Charge, 0, 1, np.int8),
                          (ZNCharge(3), 0, 2, np.int8)])
def test_Charge_matmul(chargetype, B0, B1, dtype):
  D = 1000
  np.random.seed(10)
  C1 = np.random.randint(B0, B1 + 1, D).astype(dtype)
  C2 = np.random.randint(B0, B1 + 1, D).astype(dtype)
  C3 = np.random.randint(B0, B1 + 1, D).astype(dtype)

  q1 = chargetype(C1)
  q2 = chargetype(C2)
  q3 = chargetype(C3)

  Q = q1 @ q2 @ q3
  Q_ = BaseCharge([C1, C2, C3], charge_types=[[chargetype] for _ in range(3)])
  np.testing.assert_allclose(np.stack(Q.charges), np.stack(Q_.charges))

  
def test_BaseCharge_matmul_raises():
  B = 5
  np.random.seed(10)
  C1 = np.random.randint(-B // 2, B // 2 + 1, 10).astype(np.int16)
  C2 = np.random.randint(-B // 2, B // 2 + 1, 11).astype(np.int16)

  q1 = U1Charge(C1)
  q2 = U1Charge(C2)
  with pytest.raises(ValueError):
    q1 @ q2


@pytest.mark.parametrize('chargetype, B0, B1, identity',
                         [(U1Charge, -5, 5, 0), (Z2Charge, 0, 1, 0),
                          (ZNCharge(5), 0, 4, 0), (ZNCharge(7), 0, 6, 0)])
def test_Charge_identity(chargetype, B0, B1, identity):
  D = 100
  np.random.seed(10)
  C1 = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  C2 = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  C3 = np.random.randint(B0, B1 + 1, D).astype(np.int16)

  q1 = chargetype(C1)
  q2 = chargetype(C2)
  q3 = chargetype(C3)

  Q = q1 @ q2 @ q3
  eye = Q.identity_charges(dim=2)
  np.testing.assert_allclose(eye.charges[0], identity)
  assert eye.num_symmetries == 3
  assert len(eye) == 2


@pytest.mark.parametrize("n", list(range(2, 20)))
def test_zncharge_dual_invariant(n):
  D = 100
  np.random.seed(10)
  charges = np.random.randint(0, n, D).astype(np.int16)
  a = ZNCharge(n)(charges)
  b = a.dual(True)
  np.testing.assert_allclose((b.charges[0] + a.charges[0]) % n, np.zeros((D)))


@pytest.mark.parametrize('chargetype, B0, B1, sign', [(U1Charge, -5, 5, -1),
                                                      (Z2Charge, 0, 1, 1)])
def test_Charge_mul(chargetype, B0, B1, sign):
  D = 100
  np.random.seed(10)
  C1 = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  C2 = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  q1 = chargetype(C1)
  q2 = chargetype(C2)
  q = q1 @ q2
  res = q * True
  res.expand_charge_types()
  np.testing.assert_allclose(
      np.stack(res.charges, axis=0), sign * np.stack([C1, C2], axis=0))


@pytest.mark.parametrize('n', list(range(2, 12)))
def test_Charge_mul_zncharge(n):
  chargetype = ZNCharge(n)
  D = 100
  np.random.seed(10)
  C1 = np.random.randint(0, n, D).astype(np.int16)
  C2 = np.random.randint(0, n, D).astype(np.int16)
  q1 = chargetype(C1)
  q2 = chargetype(C2)
  q = q1 @ q2
  res = q * True
  res.expand_charge_types()
  np.testing.assert_allclose(
      np.stack(res.charges), (n - np.stack([C1, C2])) % n)


def test_fuse_charges():
  num_charges = 5
  B = 6
  D = 10
  np_charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(num_charges)
  ]
  charges = [U1Charge(c) for c in np_charges]
  flows = [True, False, True, False, True]
  np_flows = np.ones(5, dtype=np.int16)
  np_flows[flows] = -1
  fused = fuse_charges(charges, flows)
  np_fused = fuse_ndarrays([c * f for c, f in zip(np_charges, np_flows)])
  np.testing.assert_allclose(fused.charges[0], np_fused)


def test_fuse_charges_raises():
  num_charges = 5
  B = 6
  D = 10
  np_charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(num_charges)
  ]
  charges = [U1Charge(c) for c in np_charges]
  flows = [True, False, True, False]
  with pytest.raises(ValueError):
    fuse_charges(charges, flows)


def test_reduce():
  q = [
      np.array([0, 1, 2, 0, 6, 1, -9, 0, -7], dtype=np.int16),
      np.array([2, 3, 4, -1, 4, 3, 1, 2, 0], dtype=np.int16)
  ]
  Q = BaseCharge(charges=q)
  targets = [
      np.array([0, 1, 6, -12], dtype=np.int16),
      np.array([2, 3, 4, 16], dtype=np.int16)
  ]
  target_charge = BaseCharge(charges=targets)

  expected = np.array([[0, 1, 6, 1, 0], [2, 3, 4, 3, 2]])
  res, locs = Q.reduce(
      target_charge, return_locations=True, return_type='charges')
  res.expand_charge_types()
  np.testing.assert_allclose(np.stack(res.charges, axis=0), expected)
  np.testing.assert_allclose(locs, [0, 1, 4, 5, 7])


def test_getitem():
  q1 = np.array([0, 1, 2, 0, 6, 1, -9, 0, -7], dtype=np.int16)
  q2 = np.array([2, 3, 4, -1, 4, 3, 1, 2, 0], dtype=np.int16)
  Q1 = U1Charge(charges=q1)
  Q2 = U1Charge(charges=q2)
  Q = Q1 @ Q2
  t1 = Q[5]
  t1.expand_charge_types()
  np.testing.assert_allclose(t1.charges, [[1], [3]])
  assert np.all([t1.charge_types[n][0] is U1Charge for n in range(2)])
  t2 = Q[[2, 5, 7]]
  t2.expand_charge_types()
  assert np.all([t2.charge_types[n][0] == U1Charge for n in range(2)])
  np.testing.assert_allclose(t2.charges, [[2, 1, 0], [4, 3, 2]])
  t3 = Q[[5, 2, 7]]
  t3.expand_charge_types()
  assert np.all([t3.charge_types[n][0] == U1Charge for n in range(2)])
  np.testing.assert_allclose(t3.charges, [[1, 2, 0], [3, 4, 2]])


def test_isin():
  np.random.seed(10)
  c1 = U1Charge(np.random.randint(-5, 5, 1000, dtype=np.int16))
  c2 = U1Charge(np.random.randint(-5, 5, 1000, dtype=np.int16))

  c = c1 @ c2
  c3 = U1Charge(np.array([-1, 0, 1], dtype=np.int16)) @ U1Charge(
      np.array([-1, 0, 1], dtype=np.int16))
  expected = np.array([[-1, 0, 1], [-1, 0, 1]])

  n = c.isin(c3)
  for m in np.nonzero(n)[0]:
    charges = c[m]
    charges.expand_charge_types()
    cs = np.stack(charges.charges, axis=0)
    #pylint: disable=unsubscriptable-object
    assert np.any([
        np.array_equal(cs[:, 0], expected[:, k])
        for k in range(expected.shape[1])
    ])
  for m in np.nonzero(np.logical_not(n))[0]:
    charges = c[m]
    charges.expand_charge_types()
    cs = np.stack(charges.charges, axis=0)
    #pylint: disable=unsubscriptable-object
    assert not np.any([
        np.array_equal(cs[:, 0], expected[:, k])
        for k in range(expected.shape[1])
    ])


def test_eq_1():
  np.random.seed(10)
  c1 = U1Charge(np.array([-2, -1, 0, 1, -1, 3, 4, 5], dtype=np.int16))
  c2 = U1Charge(np.array([-1, 0, 1, 2, 0, 4, 5, 6], dtype=np.int16))
  c = c1 @ c2
  c3 = U1Charge(np.array([-1], dtype=np.int16)) @ U1Charge(
      np.array([0], dtype=np.int16))
  exp = np.array([[-1], [0]])
  inds = np.nonzero(c == c3)[0]
  np.testing.assert_allclose(inds, [1, 4])
  for i in inds:
    tmp = c[i]
    tmp.expand_charge_types()
    np.array_equal(np.stack(tmp.charges), exp)


def test_eq_2():
  np.random.seed(10)
  c1 = U1Charge(np.array([-2, -1, 0, 1, -1, 3, 4, 5, 1], dtype=np.int16))
  c2 = U1Charge(np.array([-1, 0, 1, 2, 0, 4, 5, 6, 2], dtype=np.int16))
  c = c1 @ c2
  c3 = U1Charge(np.array([-1, 1], dtype=np.int16)) @ U1Charge(
      np.array([0, 2], dtype=np.int16))
  exp = np.array([[-1, 1], [0, 2]])
  inds = np.nonzero(c == c3)

  np.testing.assert_allclose(inds[0][inds[1] == 0], [1, 4])
  np.testing.assert_allclose(inds[0][inds[1] == 1], [3, 8])
  for i, j in zip(inds[0], inds[1]):
    tmp = c[i]
    tmp.expand_charge_types()
    np.array_equal(np.stack(tmp.charges), exp[:, j])


def test_empty():
  num_charges = 4
  charges = BaseCharge([
      np.random.randint(-5, 6, 0).astype(np.int16) for _ in range(num_charges)
  ])
  assert len(charges) == 0


def test_init_raises():
  charge_types = [[U1Charge, Z2Charge], [U1Charge]]
  with pytest.raises(ValueError):
    BaseCharge(
        [np.random.randint(-5, 6, 10).astype(np.int16) for _ in range(2)],
        charge_types=charge_types)


def test_zncharge_raises():
  with pytest.raises(ValueError, match="n must be >= 2, found 0"):
    ZNCharge(0)
  with pytest.raises(ValueError, match="Z7 charges must be in"):
    ZNCharge(7)([0, 4, 9])
