import numpy as np
# pylint: disable=line-too-long
from tensornetwork.block_sparse.index import Index, fuse_index_pair, fuse_indices
from tensornetwork.block_sparse.charge import U1Charge, BaseCharge


def test_index_fusion_mul():
  D = 10
  B = 4
  dtype = np.int16
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  charges = [q1, q2]
  i1 = Index(charges=q1, flow=False, name='index1')  #index on leg 1
  i2 = Index(charges=q2, flow=False, name='index2')  #index on leg 2

  i12 = i1 * i2
  for n in range(len(i12.charges.charges)):
    assert np.all(i12._charges[n].charges == charges[n].charges)
  assert np.all(i12.charges.charges == (q1 + q2).charges)


def test_fuse_indices():
  D = 10
  B = 4
  dtype = np.int16
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  charges = [q1, q2]
  i1 = Index(charges=q1, flow=False, name='index1')  #index on leg 1
  i2 = Index(charges=q2, flow=False, name='index2')  #index on leg 2

  i12 = fuse_indices([i1, i2])
  for n in range(len(i12.charges.charges)):
    assert np.all(i12._charges[n].charges == charges[n].charges)
  assert np.all(i12.charges.charges == (q1 + q2).charges)


def test_copy():
  D = 10
  B = 4
  dtype = np.int16
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1

  i1 = Index(charges=q1, flow=False, name='index1')
  i2 = Index(charges=q2, flow=False, name='index2')
  i3 = Index(charges=q1, flow=True, name='index3')
  i4 = Index(charges=q2, flow=True, name='index4')

  i12 = i1 * i2
  i34 = i3 * i4
  i1234 = i12 * i34
  i1234_copy = i1234.copy()

  flat1234 = i1234_copy.flat_charges
  assert flat1234[0] is not i1.flat_charges[0]
  assert flat1234[1] is not i2.flat_charges[0]
  assert flat1234[2] is not i3.flat_charges[0]
  assert flat1234[3] is not i4.flat_charges[0]
