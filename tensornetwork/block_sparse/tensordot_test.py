import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import U1Charge, charge_equal, BaseCharge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor, tensordot, outerproduct
from tensornetwork import ncon

np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def get_contractable_tensors(R1, R2, cont, dtype, num_charges, DsA, Dscomm,
                             DsB):
  assert R1 >= cont
  assert R2 >= cont
  chargesA = [
      BaseCharge(
          np.random.randint(-5, 5, (num_charges, DsA[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R1 - cont)
  ]
  commoncharges = [
      BaseCharge(
          np.random.randint(-5, 5, (num_charges, Dscomm[n])),
          charge_types=[U1Charge] * num_charges) for n in range(cont)
  ]
  chargesB = [
      BaseCharge(
          np.random.randint(-5, 5, (num_charges, DsB[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R2 - cont)
  ]
  #contracted indices
  indsA = np.random.choice(np.arange(R1), cont, replace=False)
  indsB = np.random.choice(np.arange(R2), cont, replace=False)

  flowsA = np.full(R1, False, dtype=np.bool)
  flowsB = np.full(R2, False, dtype=np.bool)
  flowsB[indsB] = True

  indicesA = [None for _ in range(R1)]
  indicesB = [None for _ in range(R2)]
  for n, ia in enumerate(indsA):
    indicesA[ia] = Index(commoncharges[n], flowsA[ia])
    indicesB[indsB[n]] = Index(commoncharges[n], flowsB[indsB[n]])
  compA = list(set(np.arange(R1)) - set(indsA))
  compB = list(set(np.arange(R2)) - set(indsB))

  for n, ca in enumerate(compA):
    indicesA[ca] = Index(chargesA[n], flowsA[ca])
  for n, cb in enumerate(compB):
    indicesB[cb] = Index(chargesB[n], flowsB[cb])
  indices_final = []
  for n in sorted(compA):
    indices_final.append(indicesA[n])
  for n in sorted(compB):
    indices_final.append(indicesB[n])
  A = BlockSparseTensor.random(indices=indicesA, dtype=dtype)
  B = BlockSparseTensor.random(indices=indicesB, dtype=dtype)
  return A, B, indsA, indsB


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_legs', [1, 2, 3, 4])
@pytest.mark.parametrize('num_charges', [1, 2])
def test_outerproduct(dtype, num_legs, num_charges):
  np.random.seed(10)
  Ds1 = np.arange(2, 2 + num_legs)
  Ds2 = np.arange(2 + num_legs, 2 + 2 * num_legs)
  is1 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds1[n])),
              charge_types=[U1Charge] * num_charges), False)
      for n in range(num_legs)
  ]
  is2 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds2[n])),
              charge_types=[U1Charge] * num_charges), False)
      for n in range(num_legs)
  ]
  a = BlockSparseTensor.random(is1, dtype=dtype)
  b = BlockSparseTensor.random(is2, dtype=dtype)
  abdense = ncon([a.todense(), b.todense()], [
      -np.arange(1, num_legs + 1, dtype=np.int16),
      -num_legs - np.arange(1, num_legs + 1, dtype=np.int16)
  ])
  ab = outerproduct(a, b)
  np.testing.assert_allclose(ab.todense(), abdense)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_legs', [2, 3])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_outerproduct_transpose(dtype, num_legs, num_charges):
  np.random.seed(10)
  Ds1 = np.arange(2, 2 + num_legs)
  Ds2 = np.arange(2 + num_legs, 2 + 2 * num_legs)
  is1 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds1[n])),
              charge_types=[U1Charge] * num_charges), False)
      for n in range(num_legs)
  ]
  is2 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds2[n])),
              charge_types=[U1Charge] * num_charges), False)
      for n in range(num_legs)
  ]
  o1 = np.arange(num_legs)
  o2 = np.arange(num_legs)
  np.random.shuffle(o1)
  np.random.shuffle(o2)
  a = BlockSparseTensor.random(is1, dtype=dtype).transpose(o1)
  b = BlockSparseTensor.random(is2, dtype=dtype).transpose(o2)

  abdense = ncon([a.todense(), b.todense()], [
      -np.arange(1, num_legs + 1, dtype=np.int16),
      -num_legs - np.arange(1, num_legs + 1, dtype=np.int16)
  ])
  ab = outerproduct(a, b)
  np.testing.assert_allclose(ab.todense(), abdense)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_legs', [2, 3])
@pytest.mark.parametrize('num_charges', [1, 2])
def test_outerproduct_transpose_reshape(dtype, num_legs, num_charges):
  np.random.seed(10)
  Ds1 = np.arange(2, 2 + num_legs)
  Ds2 = np.arange(2 + num_legs, 2 + 2 * num_legs)
  is1 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds1[n])),
              charge_types=[U1Charge] * num_charges), False)
      for n in range(num_legs)
  ]
  is2 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds2[n])),
              charge_types=[U1Charge] * num_charges), False)
      for n in range(num_legs)
  ]
  o1 = np.arange(num_legs)
  o2 = np.arange(num_legs)
  np.random.shuffle(o1)
  np.random.shuffle(o2)
  a = BlockSparseTensor.random(is1, dtype=dtype).transpose(o1)
  b = BlockSparseTensor.random(is2, dtype=dtype).transpose(o2)
  a = a.reshape([np.prod(a.shape)])
  b = b.reshape([np.prod(b.shape)])

  abdense = ncon([a.todense(), b.todense()], [[-1], [-2]])
  ab = outerproduct(a, b)
  assert ab.ndim == 2
  np.testing.assert_allclose(ab.todense(), abdense)


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2, cont", [(4, 4, 2), (4, 3, 3), (3, 4, 3)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot(R1, R2, cont, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(5, 10, R1 - cont)
  Dscomm = np.random.randint(5, 10, cont)
  DsB = np.random.randint(5, 10, R2 - cont)
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, cont, dtype,
                                                num_charges, DsA, Dscomm, DsB)
  res = tensordot(A, B, (indsA, indsB))
  dense_res = np.tensordot(A.todense(), B.todense(), (indsA, indsB))
  np.testing.assert_allclose(dense_res, res.todense())
  free_inds_A = np.sort(list(set(np.arange(len(A.shape))) - set(indsA)))
  free_inds_B = np.sort(list(set(np.arange(len(B.shape))) - set(indsB)))
  for n, fiA in enumerate(free_inds_A):
    assert charge_equal(res.charges[n][0], A.charges[fiA][0])
  for n in range(len(free_inds_A), len(free_inds_A) + len(free_inds_B)):
    assert charge_equal(res.charges[n][0],
                        B.charges[free_inds_B[n - len(free_inds_A)]][0])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_empty_tensors(dtype, num_charges):
  A, B, iA, iB = get_contractable_tensors(
      R1=4,
      R2=4,
      cont=2,
      dtype=dtype,
      num_charges=num_charges,
      DsA=[10, 0],
      Dscomm=[0, 4],
      DsB=[8, 0])
  free_inds_A = np.sort(list(set(np.arange(len(A.shape))) - set(iA)))
  free_inds_B = np.sort(list(set(np.arange(len(B.shape))) - set(iB)))
  res = tensordot(A, B, (iA, iB))
  assert len(res.data) == 0
  for n in range(2):
    assert charge_equal(res.charges[n][0], A.charges[free_inds_A[n]][0])
  for n in range(2, 4):
    assert charge_equal(res.charges[n][0], B.charges[free_inds_B[n - 2]][0])


def test_tensordot_raises():
  R1 = 3
  R2 = 3
  dtype = np.float64
  np.random.seed(10)
  Ds1 = np.arange(2, 2 + R1)
  Ds2 = np.arange(2 + R1, 2 + R1 + R2)
  is1 = [
      Index(U1Charge.random(dimension=Ds1[n], minval=-5, maxval=5), False)
      for n in range(R1)
  ]
  is2 = [
      Index(U1Charge.random(dimension=Ds2[n], minval=-5, maxval=5), False)
      for n in range(R2)
  ]
  A = BlockSparseTensor.random(is1, dtype=dtype)
  B = BlockSparseTensor.random(is2, dtype=dtype)
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 1, 2, 3], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 1], [0, 1, 2, 3]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, [0, [1, 2]])
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 0], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 1], [1, 1]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 4], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 4], [1, 4]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 1], [0, 1]))
  with pytest.raises(ValueError):
    tensordot(A, A, ([0, 1], [0, 1]))
  with pytest.raises(ValueError):
    tensordot(A, A.conj(), ([0, 1], [1, 0]))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_reshape(dtype, num_charges):
  np.random.seed(10)
  R1 = 4
  R2 = 4

  q = np.random.randint(-5, 5, (num_charges, 10), dtype=np.int16)
  charges1 = [
      BaseCharge(q, charge_types=[U1Charge] * num_charges) for n in range(R1)
  ]
  charges2 = [
      BaseCharge(q, charge_types=[U1Charge] * num_charges) for n in range(R2)
  ]
  flowsA = np.asarray([False] * R1)
  flowsB = np.asarray([True] * R2)
  A = BlockSparseTensor.random(
      indices=[Index(charges1[n], flowsA[n]) for n in range(R1)], dtype=dtype)
  B = BlockSparseTensor.random(
      indices=[Index(charges2[n], flowsB[n]) for n in range(R2)], dtype=dtype)

  Adense = A.todense().reshape((10, 10 * 10, 10))
  Bdense = B.todense().reshape((10 * 10, 10, 10))

  A = A.reshape((10, 10 * 10, 10))
  B = B.reshape((10 * 10, 10, 10))

  res = tensordot(A, B, ([0, 1], [2, 0]))
  dense = np.tensordot(Adense, Bdense, ([0, 1], [2, 0]))
  np.testing.assert_allclose(dense, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (3, 3), (4, 4), (1, 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_inner(R1, R2, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(3, 5, R1)
  Dscomm = np.random.randint(3, 5, 0)
  DsB = np.random.randint(3, 5, R2)
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, 0, dtype, num_charges,
                                                DsA, Dscomm, DsB)
  res = tensordot(A, B, (indsA, indsB))
  dense_res = np.tensordot(A.todense(), B.todense(), (indsA, indsB))
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (3, 3), (4, 4), (1, 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_inner_transpose(R1, R2, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(3, 5, R1)
  Dscomm = np.random.randint(3, 5, 0)
  DsB = np.random.randint(3, 5, R2)
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, 0, dtype, num_charges,
                                                DsA, Dscomm, DsB)
  orderA = np.arange(R1)
  orderB = np.arange(R2)
  np.random.shuffle(orderA)
  np.random.shuffle(orderB)
  A_ = A.transpose(orderA)
  B_ = B.transpose(orderB)
  _, indposA = np.unique(orderA, return_index=True)
  _, indposB = np.unique(orderB, return_index=True)
  indsA_ = indposA[indsA]
  indsB_ = indposB[indsB]
  res = tensordot(A_, B_, (indsA_, indsB_))
  dense_res = np.tensordot(A_.todense(), B_.todense(), (indsA_, indsB_))
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (2, 1), (1, 2), (1, 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_outer(R1, R2, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(3, 5, R1)
  Dscomm = np.random.randint(3, 5, 0)
  DsB = np.random.randint(3, 5, R2)
  A, B, _, _ = get_contractable_tensors(R1, R2, 0, dtype, num_charges, DsA,
                                        Dscomm, DsB)
  res = tensordot(A, B, axes=0)
  dense_res = np.tensordot(A.todense(), B.todense(), axes=0)
  np.testing.assert_allclose(dense_res, res.todense())
  for n in range(R1):
    assert charge_equal(res.charges[n][0], A.charges[n][0])
  for n in range(R1, R1 + R2):
    assert charge_equal(res.charges[n][0], B.charges[n - R1][0])
