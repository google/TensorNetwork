import numpy as np
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index, fuse_charges, fuse_degeneracies, fuse_charge_pair


def test_fuse_charge_pair():
  q1 = np.asarray([0, 1])
  q2 = np.asarray([2, 3, 4])
  fused_charges = fuse_charge_pair(q1, 1, q2, 1)
  assert np.all(fused_charges == np.asarray([2, 3, 3, 4, 4, 5]))
  fused_charges = fuse_charge_pair(q1, 1, q2, -1)
  assert np.all(fused_charges == np.asarray([-2, -1, -3, -2, -4, -3]))


def test_index_fusion_mul():
  D = 100
  B = 4
  dtype = np.int16
  q1 = np.random.randint(-B // 2, B // 2 + 1,
                         D).astype(dtype)  #quantum numbers on leg 1
  q2 = np.random.randint(-B // 2, B // 2 + 1,
                         D).astype(dtype)  #quantum numbers on leg 2
  i1 = Index(charges=q1, flow=1, name='index1')  #index on leg 1
  i2 = Index(charges=q2, flow=1, name='index2')  #index on leg 2

  i12 = i1 * i2
  assert i12.left_child is i1
  assert i12.right_child is i2
  assert np.all(i12.charges == fuse_charge_pair(q1, 1, q2, 1))


def test_index_fusion():
  D = 100
  B = 4
  dtype = np.int16
  q1 = np.random.randint(-B // 2, B // 2 + 1,
                         D).astype(dtype)  #quantum numbers on leg 1
  q2 = np.random.randint(-B // 2, B // 2 + 1,
                         D).astype(dtype)  #quantum numbers on leg 2
  i1 = Index(charges=q1, flow=1, name='index1')  #index on leg 1
  i2 = Index(charges=q2, flow=1, name='index2')  #index on leg 2

  i12 = fuse_index_pair(i1, i2)
  assert i12.left_child is i1
  assert i12.right_child is i2
  assert np.all(i12.charges == fuse_charge_pair(q1, 1, q2, 1))


def test_elementary_indices():
  D = 10
  B = 4
  dtype = np.int16
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
  q2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
  i1 = Index(charges=q1, flow=1, name='index1')
  i2 = Index(charges=q2, flow=1, name='index2')
  i3 = Index(charges=q1, flow=1, name='index3')
  i4 = Index(charges=q2, flow=1, name='index4')

  i12 = i1 * i2
  i34 = i3 * i4
  elmt12 = i12.get_elementary_indices()
  assert elmt12[0] is i1
  assert elmt12[1] is i2

  i1234 = i12 * i34
  elmt1234 = i1234.get_elementary_indices()
  assert elmt1234[0] is i1
  assert elmt1234[1] is i2
  assert elmt1234[2] is i3
  assert elmt1234[3] is i4


def test_copy():
  D = 10
  B = 4
  dtype = np.int16
  q1 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
  q2 = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
  i1 = Index(charges=q1, flow=1, name='index1')
  i2 = Index(charges=q2, flow=1, name='index2')
  i3 = Index(charges=q1, flow=-1, name='index3')
  i4 = Index(charges=q2, flow=-1, name='index4')

  i12 = i1 * i2
  i34 = i3 * i4
  i1234 = i12 * i34
  i1234_copy = i1234.copy()

  elmt1234 = i1234_copy.get_elementary_indices()
  assert elmt1234[0] is not i1
  assert elmt1234[1] is not i2
  assert elmt1234[2] is not i3
  assert elmt1234[3] is not i4
