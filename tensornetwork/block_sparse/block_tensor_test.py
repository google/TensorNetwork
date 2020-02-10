import numpy as np
import pytest

from tensornetwork.block_sparse.charge import U1Charge, fuse_charges
from tensornetwork.block_sparse.index import Index
# pylint: disable=line-too-long
from tensornetwork.block_sparse.block_tensor import compute_num_nonzero, reduce_charges, BlockSparseTensor, fuse_ndarrays, tensordot, svd, qr, diag, sqrt, trace, inv, _find_diagonal_sparse_blocks, pinv, eye, zeros, ones, randn, eigh, eig

#np_dtypes = [np.float32, np.float16, np.float64, np.complex64, np.complex128]
np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def get_contractable_tensors(R1, R2, cont, dtype):
  DsA = np.random.randint(5, 10, R1)
  DsB = np.random.randint(5, 10, R2)
  assert R1 >= cont
  assert R2 >= cont
  chargesA = [
      U1Charge(np.random.randint(-5, 5, DsA[n])) for n in range(R1 - cont)
  ]
  commoncharges = [
      U1Charge(np.random.randint(-5, 5, DsA[n + R1 - cont]))
      for n in range(cont)
  ]
  chargesB = [
      U1Charge(np.random.randint(-5, 5, DsB[n])) for n in range(R2 - cont)
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


@pytest.mark.parametrize("dtype", np_dtypes)
def test_block_sparse_init(dtype):
  D = 10  #bond dimension
  B = 10  #number of blocks
  rank = 4
  flows = np.asarray([False for _ in range(rank)])
  flows[-2::] = True
  charges = [
      U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16))
      for _ in range(rank)
  ]
  indices = [
      Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
      for n in range(rank)
  ]
  num_elements = compute_num_nonzero(charges, flows)
  A = BlockSparseTensor.random(indices=indices, dtype=dtype)
  assert A.dtype == dtype
  for r in range(rank):
    assert A.indices[r].name[0] == 'index{}'.format(r)
  assert A.shape == tuple([D] * rank)
  assert len(A.data) == num_elements


def test_reduce_charges():
  left_charges = np.asarray([-2, 0, 1, 0, 0]).astype(np.int16)
  right_charges = np.asarray([-1, 0, 2, 1]).astype(np.int16)
  target_charge = np.zeros((1, 1), dtype=np.int16)
  fused_charges = fuse_ndarrays([left_charges, right_charges])
  dense_positions = reduce_charges(
      [U1Charge(left_charges), U1Charge(right_charges)], [False, False],
      target_charge,
      return_locations=True)
  np.testing.assert_allclose(
      dense_positions[1],
      np.nonzero(fused_charges == target_charge[0, 0])[0])


@pytest.mark.parametrize("R", [2, 3, 4, 5, 6, 7])
def test_transpose(R):
  Ds = np.random.randint(3, 8, R)
  final_order = np.arange(R)
  np.random.shuffle(final_order)
  charges = [U1Charge(np.random.randint(-5, 5, Ds[n])) for n in range(R)]
  flows = np.full(R, fill_value=False, dtype=np.bool)
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(indices=indices)
  Adense = A.todense()
  dense_res = np.transpose(Adense, final_order)
  B = A.transpose(final_order)
  np.testing.assert_allclose(dense_res, B.todense())


def test_reshape():
  R = 4
  Ds = [3, 4, 5, 6]
  charges = [U1Charge(np.random.randint(-5, 5, Ds[n])) for n in range(R)]
  flows = np.full(R, fill_value=False, dtype=np.bool)
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(indices=indices)
  B = A.reshape([Ds[0] * Ds[1], Ds[2], Ds[3]])
  Adense = A.todense()
  Bdense = Adense.reshape([Ds[0] * Ds[1], Ds[2], Ds[3]])
  np.testing.assert_allclose(Bdense, B.todense())


def test_reshape_transpose():
  R = 4
  Ds = [3, 4, 5, 6]
  charges = [U1Charge(np.random.randint(-5, 5, Ds[n])) for n in range(R)]
  flows = np.full(R, fill_value=False, dtype=np.bool)
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(indices=indices)
  B = A.reshape([Ds[0] * Ds[1], Ds[2], Ds[3]]).transpose([2, 0, 1])
  dense = A.todense().reshape([Ds[0] * Ds[1], Ds[2],
                               Ds[3]]).transpose([2, 0, 1])
  np.testing.assert_allclose(dense, B.todense())


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2, cont", [(4, 4, 2), (4, 3, 3), (3, 4, 3)])
def test_tensordot(R1, R2, cont, dtype):
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, cont, dtype)
  res = tensordot(A, B, (indsA, indsB))
  dense_res = np.tensordot(A.todense(), B.todense(), (indsA, indsB))
  np.testing.assert_allclose(dense_res, res.todense())


def test_tensordot_reshape():
  R1 = 4
  R2 = 4

  q = np.random.randint(-5, 5, 10, dtype=np.int16)
  charges1 = [U1Charge(q) for n in range(R1)]
  charges2 = [U1Charge(q) for n in range(R2)]
  flowsA = np.asarray([False] * R1)
  flowsB = np.asarray([True] * R2)
  A = BlockSparseTensor.random(indices=[
      Index(charges1[n], flowsA[n], name='a{}'.format(n)) for n in range(R1)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges2[n], flowsB[n], name='b{}'.format(n)) for n in range(R2)
  ])

  Adense = A.todense().reshape((10, 10 * 10, 10))
  Bdense = B.todense().reshape((10 * 10, 10, 10))

  A = A.reshape((10, 10 * 10, 10))
  B = B.reshape((10 * 10, 10, 10))

  res = tensordot(A, B, ([0, 1], [2, 0]))
  dense = np.tensordot(Adense, Bdense, ([0, 1], [2, 0]))
  np.testing.assert_allclose(dense, res.todense())


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2, cont", [(4, 4, 2), (4, 3, 3), (3, 4, 3)])
def test_tensordot_final_order(R1, R2, cont, dtype):
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, cont, dtype)
  final_order = np.arange(R1 + R2 - 2 * cont)
  np.random.shuffle(final_order)
  res = tensordot(A, B, (indsA, indsB), final_order=final_order)
  dense_res = np.transpose(
      np.tensordot(A.todense(), B.todense(), (indsA, indsB)), final_order)
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (3, 3), (4, 4), (1, 1)])
def test_tensordot_inner(R1, R2, dtype):

  A, B, indsA, indsB = get_contractable_tensors(R1, R2, 0, dtype)
  res = tensordot(A, B, (indsA, indsB))
  dense_res = np.tensordot(A.todense(), B.todense(), (indsA, indsB))
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (2, 1), (1, 2), (1, 1)])
def test_tensordot_outer(R1, R2, dtype):
  A, B, _, _ = get_contractable_tensors(R1, R2, 0, dtype)
  res = tensordot(A, B, axes=0)
  dense_res = np.tensordot(A.todense(), B.todense(), axes=0)
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1, R2", [(2, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_svd_prod(dtype, R, R1, R2):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  A = A.reshape([D**R1, D**R2])
  U, S, V = svd(A, full_matrices=False)
  A_ = U @ diag(S) @ V
  np.testing.assert_allclose(A.data, A_.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1, R2", [(2, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_svd_singvals(dtype, R, R1, R2):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  A = A.reshape([D**R1, D**R2])
  _, S1, _ = svd(A, full_matrices=False)
  S2 = svd(A, full_matrices=False, compute_uv=False)
  np.testing.assert_allclose(S1.data, S2.data)
  Sdense = np.linalg.svd(A.todense(), compute_uv=False)
  np.testing.assert_allclose(
      np.sort(Sdense[Sdense > 1E-15]), np.sort(S2.data[S2.data > 0.0]))


@pytest.mark.parametrize("mode", ['complete', 'reduced'])
@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R, R1, R2", [(2, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_qr_prod(dtype, R, R1, R2, mode):
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  A = A.reshape([D**R1, D**R2])
  Q, R = qr(A, mode=mode)
  A_ = Q @ R
  np.testing.assert_allclose(A.data, A_.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4])
def test_sqrt(dtype, R):
  D = 20
  charges = [U1Charge.random(-3, 3, D) for n in range(R)]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  res = sqrt(A)
  res_dense = np.sqrt(A.todense())
  np.testing.assert_allclose(res.todense(), res_dense)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_trace(dtype):
  D = 20
  R = 2
  charge = U1Charge.random(-3, 3, D)
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                               dtype=dtype)
  res = trace(A)
  res_dense = np.trace(A.todense())
  np.testing.assert_allclose(res.todense(), res_dense)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye(dtype):
  D = 10
  charges = U1Charge.random(-3, 3, D)
  flow = False
  index = Index(charges, flow)
  A = eye(index, dtype=dtype)
  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      A.flat_charges, A.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(A.data[block], shapes[:, n])
    np.testing.assert_almost_equal(t, np.eye(t.shape[0], t.shape[1]))


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_2(dtype):
  D = 10
  charges = U1Charge.random(-3, 3, D)
  index0, index1 = Index(charges, False), Index(charges, True)
  A = eye(index0, index1, dtype=dtype)
  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      A.flat_charges, A.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(A.data[block], shapes[:, n])
    np.testing.assert_almost_equal(t, np.eye(t.shape[0], t.shape[1]))


@pytest.mark.parametrize("dtype",
                         [np.float32, np.float64, np.complex64, np.complex128])
def test_rand(dtype):
  R = 3
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(indices, boundaries=(-0.5, 0.5), dtype=dtype)
  assert np.all(A.data > -0.5)
  assert np.all(A.data < 0.5)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_zeros(dtype):
  R = 3
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.zeros(indices, dtype=dtype)
  np.testing.assert_allclose(A.data, 0.0)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_ones(dtype):
  R = 3
  D = 30
  charges = [U1Charge.random(-5, 5, D) for n in range(R)]
  flows = [True] * R
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.ones(indices, dtype=dtype)
  np.testing.assert_allclose(A.data, 1.0)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_inv(dtype):
  R = 2
  D = 10
  charges = [U1Charge.random(-3, 3, D) for n in range(R)]
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charges[0], flows[n]) for n in range(R)],
                               (-0.5, 0.5),
                               dtype=dtype)
  invA = inv(A)
  left_eye = invA @ A

  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      left_eye.flat_charges, left_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(left_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12

  right_eye = A @ invA
  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      right_eye.flat_charges, right_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(right_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_pinv(dtype):
  R = 2
  D = 10
  charges = [U1Charge.random(-3, 3, D) for n in range(R)]
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charges[0], flows[n]) for n in range(R)],
                               (-0.5, 0.5),
                               dtype=dtype)
  invA = pinv(A)
  left_eye = invA @ A

  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      left_eye.flat_charges, left_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(left_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12

  right_eye = A @ invA
  blocks, charges, shapes = _find_diagonal_sparse_blocks(
      right_eye.flat_charges, right_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(right_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [1, 2, 3])
def test_eigh_prod(dtype, R):
  D = 10
  charge = U1Charge.random(-5, 5, D)
  flows = [True] * R + [False] * R
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(2 * R)],
                               dtype=dtype)
  A = A.reshape([D**R, D**R])
  B = A + A.T.conj()
  E, V = eigh(B)
  B_ = V @ diag(E) @ V.conj().T
  np.testing.assert_allclose(B.data, B_.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [1, 2, 3])
def test_eig_prod(dtype, R):
  D = 10
  charge = U1Charge.random(-5, 5, D)
  flows = [True] * R + [False] * R
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(2 * R)],
                               dtype=dtype)
  A = A.reshape([D**R, D**R])
  E, V = eig(A)
  A_ = V @ diag(E) @ inv(V)
  np.testing.assert_allclose(A.data, A_.data)
