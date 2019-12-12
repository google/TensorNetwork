import numpy as np
# pylint: disable=line-too-long
from tensornetwork.block_tensor.index import Index, fuse_index_pair, split_index, fuse_charges, fuse_degeneracies


def test_fuse_charges():
  q1 = np.asarray([0, 1])
  q2 = np.asarray([2, 3, 4])
  fused_charges = fuse_charges(q1, 1, q2, 1)
  assert np.all(fused_charges == np.asarray([2, 3, 3, 4, 4, 5]))
  fused_charges = fuse_charges(q1, 1, q2, -1)
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
  assert np.all(i12.charges == fuse_charges(q1, 1, q2, 1))


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
  assert np.all(i12.charges == fuse_charges(q1, 1, q2, 1))
