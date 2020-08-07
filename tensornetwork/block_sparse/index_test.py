import numpy as np
from tensornetwork.block_sparse.index import (Index, fuse_index_pair,
                                              fuse_indices)
from tensornetwork.block_sparse.charge import (U1Charge, BaseCharge,
                                               fuse_charges)
import pytest

def test_index():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  i = Index(charges=[q1, q2], flow=[False, True])
  assert len(i) == D**2
  assert i.dim == D**2


def test_index_eq():
  q1 = U1Charge(np.array([-1, -2, 0, 8, 7]))
  q2 = U1Charge(np.array([-1, -2, 0, 8, 7]))
  q3 = U1Charge(np.array([-1, 0, 8, 7]))
  i1 = Index(charges=q1, flow=False)
  i2 = Index(charges=q2, flow=False)
  i3 = Index(charges=q3, flow=False)
  i4 = Index(charges=[q1, q2], flow=[False, True])
  i5 = Index(charges=[q1, q2], flow=[False, True])
  i6 = Index(charges=[q1, q2], flow=[False, False])
  assert i1 == i2
  assert i1 != i3
  assert i1 != i4
  assert i4 == i5
  assert i5 != i6


def test_index_flip_flow():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  i = Index(charges=[q1, q2], flow=[False, True])
  np.testing.assert_allclose(i.flip_flow().flow, [True, False])


def test_index_charges():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  i = Index(charges=[q1, q2], flow=[False, True])
  fused = fuse_charges([q1, q2], [False, True])
  np.testing.assert_allclose(i.charges.charges, fused.charges)


def test_index_fusion_mul():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  charges = [q1, q2]
  i1 = Index(charges=q1, flow=False)  #index on leg 1
  i2 = Index(charges=q2, flow=False)  #index on leg 2

  i12 = i1 * i2
  for n in range(i12.charges.charges.shape[1]):
    assert np.all(i12._charges[n].charges == charges[n].charges)
  assert np.all(i12.charges.charges == (q1 + q2).charges)


def test_fuse_indices():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  charges = [q1, q2]
  i1 = Index(charges=q1, flow=False)  #index on leg 1
  i2 = Index(charges=q2, flow=False)  #index on leg 2

  i12 = fuse_indices([i1, i2])
  for n in range(i12.charges.charges.shape[1]):
    assert np.all(i12._charges[n].charges == charges[n].charges)
  assert np.all(i12.charges.charges == (q1 + q2).charges)


def test_index_copy():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  i = Index(charges=[q1, q2], flow=[False, True])
  icopy = i.copy()
  assert not np.any([a is b for a, b in zip(i._charges, icopy._charges)])
  assert i.flow is not icopy.flow


def test_index_copy_2():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1,
                                  D).astype(dtype))  #quantum numbers on leg 1

  i1 = Index(charges=q1, flow=False)
  i2 = Index(charges=q2, flow=False)
  i3 = Index(charges=q1, flow=True)
  i4 = Index(charges=q2, flow=True)

  i12 = i1 * i2
  i34 = i3 * i4
  i1234 = i12 * i34
  i1234_copy = i1234.copy()

  flat1234 = i1234_copy.flat_charges
  assert flat1234[0] is not i1.flat_charges[0]
  assert flat1234[1] is not i2.flat_charges[0]
  assert flat1234[2] is not i3.flat_charges[0]
  assert flat1234[3] is not i4.flat_charges[0]

def test_index_raises():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  with pytest.raises(TypeError):
    Index(charges=[q1, q2], flow=[2, True])

def test_repr():
  D = 10
  B = 4
  dtype = np.int16
  np.random.seed(10)
  q1 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  q2 = U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype))
  index = Index(charges=[q1, q2], flow=[False, True])  
  dense_shape = f"Dimension: {str(index.dim)} \n"
  charge_str = str(index._charges).replace('\n,', ',\n')
  charge_str = charge_str.replace('\n', '\n            ')
  charges = f"Charges:  {charge_str} \n"
  flow_info = f"Flows:  {str(index.flow)} \n"
  res = f"Index:\n  {dense_shape}  {charges}  {flow_info} "
  assert res == index.__repr__()
