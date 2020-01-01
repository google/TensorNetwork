import numpy as np
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index, fuse_charges, fuse_degeneracies, fuse_charge_pair, fuse_indices, unfuse, U1Charge, Charge, BaseCharge


def test_U1Charge_mul():
  q = U1Charge(np.asarray([0, 1]))
  q2 = 2 * q
  q3 = q * 2
  assert np.all(q2.charges == np.asarray([0, 2]))
  assert np.all(q3.charges == np.asarray([0, 2]))


def test_U1Charge_add():
  q1 = U1Charge(np.asarray([0, 1]))
  q2 = U1Charge(np.asarray([2, 3, 4]))
  fused_charges = q1 + q2
  assert np.all(fused_charges.charges == np.asarray([2, 3, 4, 3, 4, 5]))


def test_fuse_charge_pair():
  q1 = np.asarray([0, 1])
  q2 = np.asarray([2, 3, 4])
  fused_charges = fuse_charge_pair(q1, 1, q2, 1)
  assert np.all(fused_charges == np.asarray([2, 3, 4, 3, 4, 5]))
  fused_charges = fuse_charge_pair(q1, 1, q2, -1)
  assert np.all(fused_charges == np.asarray([-2, -3, -4, -1, -2, -3]))


def test_Charge_mul():
  q = Charge([U1Charge(np.asarray([0, 1])), U1Charge(np.asarray([-2, 3]))])
  expected = [np.asarray([0, 2]), np.asarray([-4, 6])]
  q2 = 2 * q
  q3 = q * 2
  for n in range(len(q.charges)):
    np.testing.assert_allclose(expected[n], q2.charges[n].charges)
    np.testing.assert_allclose(expected[n], q3.charges[n].charges)


def test_Charge_add():
  q1 = Charge([U1Charge(np.asarray([0, 1])), U1Charge(np.asarray([-2, 3]))])
  q2 = Charge([U1Charge(np.asarray([2, 3])), U1Charge(np.asarray([-1, 4]))])
  expected = [np.asarray([2, 3, 3, 4]), np.asarray([-3, 2, 2, 7])]
  q12 = q1 + q2
  for n in range(len(q12.charges)):
    np.testing.assert_allclose(expected[n], q12.charges[n].charges)


def test_Charge_product():
  expected = [np.asarray([0, 1]), np.asarray([2, 3])]
  q1 = U1Charge(expected[0])
  q2 = U1Charge(expected[1])
  prod = q1 @ q2
  for n in range(len(prod.charges)):
    np.testing.assert_allclose(prod.charges[n].charges, expected[n])

  B = 4
  dtype = np.int16
  D = 10
  Q1 = Charge(charges=[
      U1Charge(charges=np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
      for _ in range(2)
  ])
  Q2 = Charge(charges=[
      U1Charge(charges=np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
      for _ in range(2)
  ])
  prod = Q1 @ Q2
  expected = Q1.charges + Q2.charges
  for n in range(len(prod.charges)):
    np.testing.assert_allclose(prod.charges[n].charges, expected[n].charges)
    assert isinstance(prod.charges[n], BaseCharge)

  q1 = U1Charge(charges=np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  prod = q1 @ Q2
  expected = [q1] + Q2.charges
  for n in range(len(prod.charges)):
    np.testing.assert_allclose(prod.charges[n].charges, expected[n].charges)
    assert isinstance(prod.charges[n], BaseCharge)


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


# def test_index_fusion_mul():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 1
#   q2 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 2
#   i1 = Index(charges=q1, flow=1, name='index1')  #index on leg 1
#   i2 = Index(charges=q2, flow=1, name='index2')  #index on leg 2

#   i12 = i1 * i2
#   assert i12.left_child is i1
#   assert i12.right_child is i2
#   assert np.all(i12.charges == fuse_charge_pair(q1, 1, q2, 1))

# def test_fuse_index_pair():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 1
#   q2 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 2
#   i1 = Index(charges=q1, flow=1, name='index1')  #index on leg 1
#   i2 = Index(charges=q2, flow=1, name='index2')  #index on leg 2

#   i12 = fuse_index_pair(i1, i2)
#   assert i12.left_child is i1
#   assert i12.right_child is i2
#   assert np.all(i12.charges == fuse_charge_pair(q1, 1, q2, 1))

# def test_fuse_indices():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 1
#   q2 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 2
#   i1 = Index(charges=q1, flow=1, name='index1')  #index on leg 1
#   i2 = Index(charges=q2, flow=1, name='index2')  #index on leg 2

#   i12 = fuse_indices([i1, i2])
#   assert i12.left_child is i1
#   assert i12.right_child is i2
#   assert np.all(i12.charges == fuse_charge_pair(q1, 1, q2, 1))

# def test_split_index():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 1
#   q2 = np.random.randint(-B // 2, B // 2 + 1,
#                          D).astype(dtype)  #quantum numbers on leg 2
#   i1 = Index(charges=q1, flow=1, name='index1')  #index on leg 1
#   i2 = Index(charges=q2, flow=1, name='index2')  #index on leg 2

#   i12 = i1 * i2
#   i1_, i2_ = split_index(i12)
#   assert i1 is i1_
#   assert i2 is i2_
#   np.testing.assert_allclose(q1, i1.charges)
#   np.testing.assert_allclose(q2, i2.charges)
#   np.testing.assert_allclose(q1, i1_.charges)
#   np.testing.assert_allclose(q2, i2_.charges)
#   assert i1_.name == 'index1'
#   assert i2_.name == 'index2'
#   assert i1_.flow == i1.flow
#   assert i2_.flow == i2.flow

# def test_elementary_indices():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   q2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   q3 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   q4 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   i1 = Index(charges=q1, flow=1, name='index1')
#   i2 = Index(charges=q2, flow=1, name='index2')
#   i3 = Index(charges=q3, flow=1, name='index3')
#   i4 = Index(charges=q4, flow=1, name='index4')

#   i12 = i1 * i2
#   i34 = i3 * i4
#   elmt12 = i12.get_elementary_indices()
#   assert elmt12[0] is i1
#   assert elmt12[1] is i2

#   i1234 = i12 * i34
#   elmt1234 = i1234.get_elementary_indices()
#   assert elmt1234[0] is i1
#   assert elmt1234[1] is i2
#   assert elmt1234[2] is i3
#   assert elmt1234[3] is i4
#   assert elmt1234[0].name == 'index1'
#   assert elmt1234[1].name == 'index2'
#   assert elmt1234[2].name == 'index3'
#   assert elmt1234[3].name == 'index4'
#   assert elmt1234[0].flow == i1.flow
#   assert elmt1234[1].flow == i2.flow
#   assert elmt1234[2].flow == i3.flow
#   assert elmt1234[3].flow == i4.flow

#   np.testing.assert_allclose(q1, i1.charges)
#   np.testing.assert_allclose(q2, i2.charges)
#   np.testing.assert_allclose(q3, i3.charges)
#   np.testing.assert_allclose(q4, i4.charges)

# def test_leave():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   q2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   i1 = Index(charges=q1, flow=1, name='index1')
#   i2 = Index(charges=q2, flow=1, name='index2')
#   assert i1.is_leave
#   assert i2.is_leave

#   i12 = i1 * i2
#   assert not i12.is_leave

# def test_copy():
#   D = 10
#   B = 4
#   dtype = np.int16
#   q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   q2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   i1 = Index(charges=q1, flow=1, name='index1')
#   i2 = Index(charges=q2, flow=1, name='index2')
#   i3 = Index(charges=q1, flow=-1, name='index3')
#   i4 = Index(charges=q2, flow=-1, name='index4')

#   i12 = i1 * i2
#   i34 = i3 * i4
#   i1234 = i12 * i34
#   i1234_copy = i1234.copy()

#   elmt1234 = i1234_copy.get_elementary_indices()
#   assert elmt1234[0] is not i1
#   assert elmt1234[1] is not i2
#   assert elmt1234[2] is not i3
#   assert elmt1234[3] is not i4

# def test_unfuse():
#   q1 = np.random.randint(-4, 5, 10).astype(np.int16)
#   q2 = np.random.randint(-4, 5, 4).astype(np.int16)
#   q3 = np.random.randint(-4, 5, 4).astype(np.int16)
#   q12 = fuse_charges([q1, q2], [1, 1])
#   q123 = fuse_charges([q12, q3], [1, 1])
#   nz = np.nonzero(q123 == 0)[0]
#   q12_inds, q3_inds = unfuse(nz, len(q12), len(q3))

#   q1_inds, q2_inds = unfuse(q12_inds, len(q1), len(q2))
#   np.testing.assert_allclose(q1[q1_inds] + q2[q2_inds] + q3[q3_inds],
#                              np.zeros(len(q1_inds), dtype=np.int16))
