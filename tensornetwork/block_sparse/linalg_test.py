import numpy as np
import pytest
from tensornetwork.block_sparse.charge import (U1Charge, fuse_charges,
                                               charge_equal, BaseCharge)
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import (ChargeArray,
                                                          BlockSparseTensor)
from tensornetwork.block_sparse.blocksparse_utils import _find_diagonal_sparse_blocks  #pylint: disable=line-too-long
from tensornetwork.block_sparse.utils import unique
from tensornetwork import ncon
from tensornetwork.block_sparse.linalg import (norm, diag, reshape, transpose,
                                               conj, svd, qr, eigh, eig, inv,
                                               sqrt, trace, eye, pinv, sign)
import tensornetwork.block_sparse.linalg as linalg

np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize('dtype', np_dtypes)
def test_norm(dtype):
  np.random.seed(10)
  Ds = np.asarray([8, 9, 10, 11])
  rank = Ds.shape[0]
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  dense_norm = np.linalg.norm(arr.todense())
  np.testing.assert_allclose(norm(arr), dense_norm)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('Ds', [[200, 100], [100, 200]])
@pytest.mark.parametrize('flow', [False, True])
def test_get_diag(dtype, num_charges, Ds, flow):
  np.random.seed(10)
  np_flow = -np.int((np.int(flow) - 0.5) * 2)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-2, 3, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), flow) for n in range(2)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  fused = fuse_charges(arr.flat_charges, arr.flat_flows)
  inds = np.nonzero(fused == np.zeros((1, num_charges), dtype=np.int16))[0]
  # pylint: disable=no-member
  left, _ = np.divmod(inds, Ds[1])
  unique_charges = unique(np_flow * (indices[0]._charges[0].charges[left, :]))
  diagonal = diag(arr)
  sparse_blocks, _, block_shapes = _find_diagonal_sparse_blocks(
      arr.flat_charges, arr.flat_flows, 1)
  data = np.concatenate([
      np.diag(np.reshape(arr.data[sparse_blocks[n]], block_shapes[:, n]))
      for n in range(len(sparse_blocks))
  ])
  np.testing.assert_allclose(data, diagonal.data)
  np.testing.assert_allclose(unique_charges,
                             diagonal.flat_charges[0].unique_charges)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('Ds', [[0, 100], [100, 0]])
def test_get_empty_diag(dtype, num_charges, Ds):
  np.random.seed(10)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-2, 3, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), False) for n in range(2)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  diagonal = diag(arr)
  np.testing.assert_allclose([], diagonal.data)
  for c in diagonal.flat_charges:
    assert len(c) == 0


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('flow', [False, True])
def test_create_diag(dtype, num_charges, flow):
  np.random.seed(10)
  D = 200
  index = Index(
      BaseCharge(
          np.random.randint(-2, 3, (D, num_charges)),
          charge_types=[U1Charge] * num_charges), flow)

  arr = ChargeArray.random([index], dtype=dtype)
  diagarr = diag(arr)
  dense = np.ravel(diagarr.todense())
  np.testing.assert_allclose(
      np.sort(dense[dense != 0.0]), np.sort(diagarr.data[diagarr.data != 0.0]))

  sparse_blocks, charges, block_shapes = _find_diagonal_sparse_blocks(
      diagarr.flat_charges, diagarr.flat_flows, 1)

  for n, block in enumerate(sparse_blocks):
    shape = block_shapes[:, n]
    block_diag = np.diag(np.reshape(diagarr.data[block], shape))
    np.testing.assert_allclose(
        arr.data[np.squeeze((index._charges[0] * flow) == charges[n])],
        block_diag)


def test_diag_raises():
  np.random.seed(10)
  Ds = [8, 9, 10]
  rank = len(Ds)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-2, 3, (Ds[n], 1)), charge_types=[U1Charge]),
          False) for n in range(rank)
  ]
  arr = BlockSparseTensor.random(indices)
  chargearr = ChargeArray.random([indices[0], indices[1]])
  with pytest.raises(ValueError):
    diag(arr)
  with pytest.raises(ValueError):
    diag(chargearr)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_tn_reshape(dtype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  arr2 = reshape(arr, [72, 110])
  for n in range(2):
    for m in range(2):
      assert charge_equal(arr2.charges[n][m], indices[n * 2 + m].charges)
  np.testing.assert_allclose(arr2.shape, [72, 110])
  np.testing.assert_allclose(arr2._order, [[0, 1], [2, 3]])
  np.testing.assert_allclose(arr2.flows, [[False, False], [False, False]])
  assert arr2.ndim == 2
  arr3 = reshape(arr, Ds)
  for n in range(4):
    assert charge_equal(arr3.charges[n][0], indices[n].charges)

  np.testing.assert_allclose(arr3.shape, Ds)
  np.testing.assert_allclose(arr3._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr3.flows, [[False], [False], [False], [False]])
  assert arr3.ndim == 4


def test_tn_transpose():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices)
  order = [2, 1, 0, 3]
  arr2 = transpose(arr, order)
  np.testing.assert_allclose(Ds[order], arr2.shape)
  np.testing.assert_allclose(arr2._order, [[2], [1], [0], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [False], [True], [False]])


def test_tn_transpose_reshape():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices)
  arr2 = transpose(arr, [2, 0, 1, 3])
  arr3 = reshape(arr2, [80, 99])
  np.testing.assert_allclose(arr3.shape, [80, 99])
  np.testing.assert_allclose(arr3._order, [[2, 0], [1, 3]])
  np.testing.assert_allclose(arr3.flows, [[True, True], [False, False]])

  arr4 = transpose(arr3, [1, 0])
  np.testing.assert_allclose(arr4.shape, [99, 80])
  np.testing.assert_allclose(arr4._order, [[1, 3], [2, 0]])
  np.testing.assert_allclose(arr4.flows, [[False, False], [True, True]])

  arr5 = reshape(arr4, [9, 11, 10, 8])
  np.testing.assert_allclose(arr5.shape, [9, 11, 10, 8])
  np.testing.assert_allclose(arr5._order, [[1], [3], [2], [0]])
  np.testing.assert_allclose(arr5.flows, [[False], [False], [True], [True]])


@pytest.mark.parametrize('dtype', np_dtypes)
def test_tn_conj(dtype):
  np.random.seed(10)
  indices = [
      Index(U1Charge.random(dimension=10, minval=-5, maxval=5), False)
      for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = conj(a)
  np.testing.assert_allclose(b.data, np.conj(a.data))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds, R1", [([20, 21], 1), ([18, 19, 20], 2),
                                    ([18, 19, 20], 1), ([0, 10], 1),
                                    ([10, 0], 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_svd_prod(dtype, Ds, R1, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])

  U, S, V = svd(A, full_matrices=False)
  A_ = U @ diag(S) @ V
  assert A_.dtype == A.dtype
  np.testing.assert_allclose(A.data, A_.data)
  for n in range(len(A._charges)):
    assert charge_equal(A_._charges[n], A._charges[n])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds, R1", [([20, 21], 1), ([18, 19, 20], 2),
                                    ([18, 19, 20], 1), ([0, 10], 1),
                                    ([10, 0], 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_svd_singvals(dtype, Ds, R1, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])
  _, S1, _ = svd(A, full_matrices=False)
  S2 = svd(A, full_matrices=False, compute_uv=False)
  np.testing.assert_allclose(S1.data, S2.data)
  Sdense = np.linalg.svd(A.todense(), compute_uv=False)
  np.testing.assert_allclose(
      np.sort(Sdense[Sdense > 1E-15]), np.sort(S2.data[S2.data > 0.0]))


def test_svd_raises():
  np.random.seed(10)
  dtype = np.float64
  Ds = [10, 11, 12]
  R = len(Ds)
  charges = [
      BaseCharge(np.random.randint(-5, 6, (Ds[n], 1)), charge_types=[U1Charge])
      for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  with pytest.raises(NotImplementedError):
    svd(A, full_matrices=False, compute_uv=False)


#A sanity check that does not use symmetries (all charges are 0)
def test_qr_r_mode():
  Ds = [10, 11]
  dtype = np.float64
  np.random.seed(10)
  rank = len(Ds)
  charges = [
      BaseCharge(np.zeros((Ds[n], 1)), charge_types=[U1Charge] * 1)
      for n in range(rank)
  ]
  flows = [True] * rank
  A = BlockSparseTensor.random(
      [Index(charges[n], flows[n]) for n in range(rank)], dtype=dtype)
  d1 = np.prod(Ds[:1])
  d2 = np.prod(Ds[1:])
  A = A.reshape([d1, d2])
  R = qr(A, mode='r')
  R_np = np.linalg.qr(A.todense(), mode='r')
  np.testing.assert_allclose(
      np.abs(np.diag(R.todense())), np.abs(np.diag(R_np)))


@pytest.mark.parametrize("mode", ['complete', 'reduced'])
@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds, R1", [([20, 21], 1), ([18, 19, 20], 2),
                                    ([18, 19, 20], 1), ([10, 0], 1),
                                    ([0, 10], 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_qr_prod(dtype, Ds, R1, mode, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])
  Q, R = qr(A, mode=mode)
  A_ = Q @ R
  assert A_.dtype == A.dtype
  np.testing.assert_allclose(A.data, A_.data)
  for n in range(len(A._charges)):
    assert charge_equal(A_._charges[n], A._charges[n])



def test_qr_raises():
  np.random.seed(10)
  dtype = np.float64
  num_charges = 1
  Ds = [20, 21, 22]
  R1 = 2
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  B = A.reshape([d1, d2])
  with pytest.raises(ValueError, match='unknown value'):
    qr(B, mode='fake_mode')
  with pytest.raises(NotImplementedError, match="mode `raw`"):
    qr(B, mode='raw')
  with pytest.raises(NotImplementedError, match="qr currently"):
    qr(A)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds", [[20], [9, 10], [6, 7, 8], [0]])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_eigh_prod(dtype, Ds, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(
      inds + [i.copy().flip_flow() for i in inds], dtype=dtype)
  dims = np.prod(Ds)
  A = A.reshape([dims, dims])
  B = A + A.T.conj()
  E, V = eigh(B)
  B_ = V @ diag(E) @ V.conj().T
  np.testing.assert_allclose(B.contiguous(inplace=True).data, B_.data)
  for n in range(len(B._charges)):
    assert charge_equal(B_._charges[n], B._charges[n])


def test_eigh_raises():
  np.random.seed(10)
  num_charges = 1
  D = 20
  R = 3
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(inds)
  with pytest.raises(NotImplementedError):
    eigh(A)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_inv(dtype, num_charges):
  np.random.seed(10)
  R = 2
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                               (-0.5, 0.5),
                               dtype=dtype)
  invA = inv(A)
  left_eye = invA @ A

  blocks, _, shapes = _find_diagonal_sparse_blocks(left_eye.flat_charges,
                                                   left_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(left_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12

  right_eye = A @ invA
  blocks, _, shapes = _find_diagonal_sparse_blocks(right_eye.flat_charges,
                                                   right_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(right_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12


def test_inv_raises():
  num_charges = 1
  np.random.seed(10)
  R = 3
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  A = BlockSparseTensor.random([Index(charge, False) for n in range(R)],
                               (-0.5, 0.5))
  with pytest.raises(ValueError):
    inv(A)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds", [[20], [9, 10], [6, 7, 8], [0]])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_eig_prod(dtype, Ds, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]

  A = BlockSparseTensor.random(
      inds + [i.copy().flip_flow() for i in inds], dtype=dtype)
  dims = np.prod(Ds)
  A = A.reshape([dims, dims])
  E, V = eig(A)
  A_ = V @ diag(E) @ inv(V)
  np.testing.assert_allclose(A.contiguous(inplace=True).data, A_.data)


def test_eig_raises():
  np.random.seed(10)
  num_charges = 1
  D = 20
  R = 3
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(inds)
  with pytest.raises(NotImplementedError):
    eig(A)


#Note the case num_charges=4 is most likely testing  empty tensors
@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize("Ds", [[20], [9, 10], [6, 7, 8], [9, 8, 0, 10]])
def test_sqrt(dtype, num_charges, Ds):
  np.random.seed(10)
  R = len(Ds)
  flows = np.random.choice([True, False], replace=True, size=R)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(R)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  sqrtarr = sqrt(arr)
  np.testing.assert_allclose(sqrtarr.data, np.sqrt(arr.data))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('D', [0, 10])
def test_eye(dtype, num_charges, D):
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flow = False
  index = Index(charge, flow)
  A = eye(index, dtype=dtype)
  blocks, _, shapes = _find_diagonal_sparse_blocks(A.flat_charges, A.flat_flows,
                                                   1)
  for n, block in enumerate(blocks):
    t = np.reshape(A.data[block], shapes[:, n])
    np.testing.assert_almost_equal(t, np.eye(t.shape[0], t.shape[1]))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('D', [0, 100])
def test_trace_matrix(dtype, num_charges, D):
  np.random.seed(10)
  R = 2
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  matrix = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                                    dtype=dtype)
  res = trace(matrix)
  res_dense = np.trace(matrix.todense())
  np.testing.assert_allclose(res.data, res_dense)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('D1, D2', [(10, 12), (0, 10)])
def test_trace_tensor(dtype, num_charges, D1, D2):
  np.random.seed(10)
  charge1 = BaseCharge(
      np.random.randint(-5, 6, (D1, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  charge2 = BaseCharge(
      np.random.randint(-5, 6, (D2, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  indices = [Index(charge1, False), Index(charge2, False), Index(charge1, True)]
  tensor = BlockSparseTensor.random(indices, dtype=dtype)
  res = trace(tensor, (0, 2))
  assert res.sparse_shape[0] == indices[1]
  res_dense = np.trace(tensor.todense(), axis1=0, axis2=2)
  np.testing.assert_allclose(res.todense(), res_dense)


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_trace_raises(num_charges):
  np.random.seed(10)
  D = 20
  charge1 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  A1 = BlockSparseTensor.random([Index(charge1, False)])
  with pytest.raises(ValueError, match="trace can only"):
    trace(A1)

  charge2 = BaseCharge(
      np.random.randint(-5, 6, (D + 1, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  indices = [
      Index(charge1, False),
      Index(charge2, True),
      Index(charge1, False)
  ]
  A2 = BlockSparseTensor.random(indices)
  with pytest.raises(ValueError, match="not matching"):
    trace(A2, axes=(0, 1))
  with pytest.raises(ValueError, match="non-matching flows"):
    trace(A2, axes=(0, 2))
  with pytest.raises(ValueError, match="has to be 2"):
    trace(A2, axes=(0, 1, 2))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_pinv(dtype, num_charges):
  np.random.seed(10)
  R = 2
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                               (-0.5, 0.5),
                               dtype=dtype)
  invA = pinv(A)
  left_eye = invA @ A

  blocks, _, shapes = _find_diagonal_sparse_blocks(left_eye.flat_charges,
                                                   left_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(left_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12

  right_eye = A @ invA
  blocks, _, shapes = _find_diagonal_sparse_blocks(right_eye.flat_charges,
                                                   right_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(right_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12


def test_pinv_raises():
  num_charges = 1
  np.random.seed(10)
  R = 3
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  A = BlockSparseTensor.random([Index(charge, False) for n in range(R)],
                               (-0.5, 0.5))
  with pytest.raises(ValueError):
    pinv(A)

def test_abs():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices)
  np.testing.assert_allclose(linalg.abs(arr).data, np.abs(arr.data))

def test_sign():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices)
  np.testing.assert_allclose(sign(arr).data, np.sign(arr.data))
