import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import BaseCharge, intersect, fuse_ndarrays, U1Charge, fuse_degeneracies, fuse_charges, Z2Charge, ZNCharge


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


def test_BaseCharge_unique():
  D = 3000
  B = 5
  np.random.seed(10)
  q = np.random.randint(-B // 2, B // 2 + 1, (2, D)).astype(np.int16)
  Q = BaseCharge(charges=q, charge_types=[U1Charge, U1Charge])
  expected = np.unique(
      q, return_index=True, return_inverse=True, return_counts=True, axis=1)
  actual = Q.unique(return_index=True, return_inverse=True, return_counts=True)
  assert np.all(actual[0].charges == expected[0])
  assert np.all(actual[1] == expected[1])
  assert np.all(actual[2] == expected[2])
  assert np.all(actual[3] == expected[3])


def test_BaseCharge_unique_sort():
  np.random.seed(10)
  unique = np.array([1, 0, -1])
  labels = np.random.randint(0, 3, 100)
  Q = U1Charge(charges=unique, charge_labels=labels)
  actual = Q.unique(
      return_index=True, return_inverse=True, return_counts=True, sort=False)
  np.testing.assert_allclose(actual[0].unique_charges, [[1, 0, -1]])


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
  assert np.all(q1.charges == charges)


@pytest.mark.parametrize('chargetype, B0, B1,sign', [(U1Charge, -5, 5, -1),
                                                     (Z2Charge, 0, 1, 1)])
def test_Charge_dual(chargetype, B0, B1, sign):
  D = 100
  np.random.seed(10)
  charges = np.random.randint(B0, B1 + 1, D).astype(np.int16)

  q1 = chargetype(charges)
  assert np.all(q1.dual(True).charges == sign * charges)


@pytest.mark.parametrize('n', list(range(2, 12)))
def test_Charge_dual_zncharges(n):
  chargetype = ZNCharge(n)
  D = 100
  np.random.seed(10)
  charges = np.random.randint(0, n, D).astype(np.int16)
  q1 = chargetype(charges)
  assert np.all(q1.dual(True).charges == (n - charges) % n)


def test_Z2Charge_raises():
  np.random.seed(10)
  charges = np.array([-1, 0, 1, 2])
  with pytest.raises(ValueError):
    Z2Charge(charges)


def get_charges(B0, B1, D, num_charges):
  return [
      np.random.randint(B0, B1 + 1, D).astype(np.int16)
      for _ in range(num_charges)
  ]


def fuse_many_charges(num_charges,
                      num_charge_types,
                      seed,
                      D,
                      B,
                      use_flows=False):
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
  nz_1, nz_2 = fuse_many_charges(
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


def test_BaseCharge_intersect_2():
  c1 = U1Charge(np.array([1, 0, -1]), charge_labels=np.array([2, 0, 1]))
  c2 = U1Charge(np.array([-1, 0, 1]))
  res = c1.intersect(c2)
  np.testing.assert_allclose(res.charges, [[-1, 0, 1]])


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


@pytest.mark.parametrize('chargetype, B0, B1', [(U1Charge, -5, 5),
                                                (Z2Charge, 0, 1),
                                                (ZNCharge(3), 0, 2)])
def test_Charge_matmul(chargetype, B0, B1):
  D = 1000
  np.random.seed(10)
  C1 = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  C2 = np.random.randint(B0, B1 + 1, D).astype(np.int16)
  C3 = np.random.randint(B0, B1 + 1, D).astype(np.int16)

  q1 = chargetype(C1)
  q2 = chargetype(C2)
  q3 = chargetype(C3)

  Q = q1 @ q2 @ q3
  Q_ = BaseCharge(
      np.stack([C1, C2, C3], axis=0),
      charge_labels=None,
      charge_types=[chargetype] * 3)
  assert np.all(Q.charges == Q_.charges)


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
  eye = Q.identity_charges
  np.testing.assert_allclose(eye.unique_charges, identity)
  assert eye.num_symmetries == 3


@pytest.mark.parametrize("n", list(range(2, 20)))
def test_zncharge_dual_invariant(n):
  D = 100
  np.random.seed(10)
  charges = np.random.randint(0, n, D).astype(np.int16)
  a = ZNCharge(n)(charges)
  b = a.dual(True)
  np.testing.assert_allclose((b.charges + a.charges) % n, np.zeros((1, D)))


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
  np.testing.assert_allclose(res.charges, sign * np.stack([C1, C2]))


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
  np.testing.assert_allclose(res.charges, (n - np.stack([C1, C2])) % n)


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
  np.testing.assert_allclose(np.squeeze(fused.charges), np_fused)


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
  q = np.array([[0, 1, 2, 0, 6, 1, -9, 0, -7], [2, 3, 4, -1, 4, 3, 1, 2, 0]])
  Q = BaseCharge(charges=q)
  target_charge = np.array([[0, 1, 6, -12], [2, 3, 4, 16]])
  expected = np.array([[0, 1, 6, 1, 0], [2, 3, 4, 3, 2]])
  res, locs = Q.reduce(target_charge, return_locations=True)
  np.testing.assert_allclose(res.charges, expected)
  np.testing.assert_allclose(locs, [0, 1, 4, 5, 7])


def test_getitem():
  q1 = np.array([0, 1, 2, 0, 6, 1, -9, 0, -7])
  q2 = np.array([2, 3, 4, -1, 4, 3, 1, 2, 0])
  Q1 = U1Charge(charges=q1)
  Q2 = U1Charge(charges=q2)
  Q = Q1 @ Q2
  t1 = Q[5]
  np.testing.assert_allclose(t1.charges, [[1], [3]])
  assert np.all([t1.charge_types[n] == U1Charge for n in range(2)])
  t2 = Q[[2, 5, 7]]
  assert np.all([t2.charge_types[n] == U1Charge for n in range(2)])
  np.testing.assert_allclose(t2.charges, [[2, 1, 0], [4, 3, 2]])
  t3 = Q[[5, 2, 7]]
  assert np.all([t3.charge_types[n] == U1Charge for n in range(2)])
  np.testing.assert_allclose(t3.charges, [[1, 2, 0], [3, 4, 2]])


def test_isin():
  np.random.seed(10)
  c1 = U1Charge(np.random.randint(-5, 5, 1000, dtype=np.int16))
  c2 = U1Charge(np.random.randint(-5, 5, 1000, dtype=np.int16))

  c = c1 @ c2
  c3 = np.array([[-1, 0, 1], [-1, 0, 1]])
  n = c.isin(c3)
  for m in np.nonzero(n)[0]:
    charges = c[m].charges
    #pylint: disable=unsubscriptable-object
    assert np.any(
        [np.array_equal(charges[:, 0], c3[:, k]) for k in range(c3.shape[1])])
  for m in np.nonzero(np.logical_not(n))[0]:
    charges = c[m].charges
    #pylint: disable=unsubscriptable-object
    assert not np.any(
        [np.array_equal(charges[:, 0], c3[:, k]) for k in range(c3.shape[1])])


def test_isin_2():
  np.random.seed(10)
  c1 = U1Charge(np.random.randint(-5, 5, 1000, dtype=np.int16))
  c2 = U1Charge(np.random.randint(-5, 5, 1000, dtype=np.int16))
  c = c1 @ c2
  c3 = U1Charge(np.array([-1, 0, 1])) @ U1Charge(np.array([-1, 0, 1]))
  n = c.isin(c3)
  for m in np.nonzero(n)[0]:
    charges = c[m].charges
    assert np.any([
        np.array_equal(charges[:, 0], c3.charges[:, k])
        for k in range(c3.charges.shape[1])
    ])
  for m in np.nonzero(np.logical_not(n))[0]:
    charges = c[m].charges
    assert not np.any([
        np.array_equal(charges[:, 0], c3.charges[:, k])
        for k in range(c3.charges.shape[1])
    ])


def test_isin_raises():

  class FakeCharge(BaseCharge):

    def __init__(self, charges, charge_labels=None, charge_types=None):
      super().__init__(charges, charge_labels, charge_types=[type(self)])

    @staticmethod
    def fuse(charge1, charge2) -> np.ndarray:
      return np.add.outer(charge1, charge2).ravel()

    @staticmethod
    def dual_charges(charges) -> np.ndarray:
      return charges * charges.dtype.type(-1)

    @staticmethod
    def identity_charge() -> np.ndarray:
      return np.int16(0)

    @classmethod
    def random(cls, dimension: int, minval: int, maxval: int) -> np.ndarray:
      charges = np.random.randint(minval, maxval, dimension, dtype=np.int16)
      return cls(charges=charges)

  np.random.seed(10)
  c1 = BaseCharge(
      np.random.randint(-5, 5, (2, 1000), dtype=np.int16),
      charge_labels=None,
      charge_types=[FakeCharge, FakeCharge])
  c2 = U1Charge(np.array([-1, 0, 1])) @ U1Charge(np.array([-1, 0, 1]))
  with pytest.raises(TypeError):
    c1.isin(c2)
  with pytest.raises(ValueError):
    c1.isin(np.random.randint(-2, 2, (2, 2, 2)))

  with pytest.raises(ValueError):
    c1.isin(np.random.randint(-2, 2, (3, 2)))


def test_eq_1():
  np.random.seed(10)
  c1 = U1Charge(np.array([-2, -1, 0, 1, -1, 3, 4, 5], dtype=np.int16))
  c2 = U1Charge(np.array([-1, 0, 1, 2, 0, 4, 5, 6], dtype=np.int16))
  c = c1 @ c2
  c3 = np.array([[-1], [0]])
  inds = np.nonzero(c == c3)[0]
  np.testing.assert_allclose(inds, [1, 4])
  for i in inds:
    np.array_equal(c[i].charges, c3)


def test_eq_2():
  np.random.seed(10)
  c1 = U1Charge(np.array([-2, -1, 0, 1, -1, 3, 4, 5, 1], dtype=np.int16))
  c2 = U1Charge(np.array([-1, 0, 1, 2, 0, 4, 5, 6, 2], dtype=np.int16))
  c = c1 @ c2
  c3 = np.array([[-1, 1], [0, 2]])
  inds = np.nonzero(c == c3)

  np.testing.assert_allclose(inds[0][inds[1] == 0], [1, 4])
  np.testing.assert_allclose(inds[0][inds[1] == 1], [3, 8])
  for i, j in zip(inds[0], inds[1]):
    np.array_equal(c[i].charges, c3[:, j])


def test_eq__raises():
  np.random.seed(10)
  num_charges = 2
  charge = BaseCharge(
      np.random.randint(-2, 3, (num_charges, 30)),
      charge_types=[U1Charge] * num_charges)
  with pytest.raises(ValueError):
    _ = charge == np.random.randint(-1, 1, (num_charges + 1, 2), dtype=np.int16)


def test_iter():
  np.random.seed(10)
  arr1 = np.array([-2, -1, 0, 1, -1, 3, 4, 5, 1], dtype=np.int16)
  arr2 = np.array([-1, 0, 1, 2, 0, 4, 5, 6, 2], dtype=np.int16)
  c1 = U1Charge(arr1)
  c2 = U1Charge(arr2)
  c = c1 @ c2
  m = 0
  for n in c:
    np.testing.assert_allclose(n, np.array([arr1[m], arr2[m]]))
    m += 1


def test_empty():
  num_charges = 4
  charges = BaseCharge(
      np.random.randint(-5, 6, (num_charges, 0)),
      charge_types=[U1Charge] * num_charges)
  assert len(charges) == 0


def test_init_raises():
  num_charges = 4
  with pytest.raises(ValueError):
    BaseCharge(
        np.random.randint(-5, 6, (num_charges, 10)),
        charge_types=[U1Charge] * (num_charges - 1))


def test_eq_raises():
  num_charges = 4
  c1 = BaseCharge(
      np.random.randint(-5, 6, (num_charges, 10)),
      charge_types=[U1Charge] * num_charges)
  c2 = BaseCharge(
      np.random.randint(-5, 6, (num_charges, 0)),
      charge_types=[U1Charge] * num_charges)
  npc = np.empty((num_charges, 0), dtype=np.int16)
  with pytest.raises(ValueError):
    c1 == c2
  with pytest.raises(ValueError):
    c1 == npc


def test_zncharge_raises():
  with pytest.raises(ValueError, match="n must be >= 2, found 0"):
    ZNCharge(0)
  with pytest.raises(ValueError, match="Z7 charges must be in"):
    ZNCharge(7)([0, 4, 9])
