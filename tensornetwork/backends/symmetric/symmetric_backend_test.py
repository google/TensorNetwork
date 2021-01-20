import numpy as np
import pytest
from tensornetwork.backends.symmetric import symmetric_backend
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.block_sparse.charge import (U1Charge, charge_equal,
                                               BaseCharge, fuse_charges)
from tensornetwork.block_sparse.blocksparse_utils import _find_diagonal_sparse_blocks  #pylint: disable=line-too-long
from tensornetwork.block_sparse.utils import unique
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import (tensordot,
                                                          BlockSparseTensor,
                                                          ChargeArray)
from tensornetwork.block_sparse.linalg import (transpose, sqrt, diag, trace,
                                               norm, eye, eigh, inv, eig)
from tensornetwork.block_sparse.initialization import (ones, zeros, randn,
                                                       random, randn_like)

from tensornetwork.block_sparse.caching import get_cacher, get_caching_status
from tensornetwork.ncon_interface import ncon
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS


np_randn_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_randn_dtypes + [np.complex64, np.complex128]
np_tensordot_dtypes = [np.float16, np.float64, np.complex128]

def get_matvec_tensors(D=10, M=5, seed=10, dtype=np.float64):
  np.random.seed(seed)
  mpsinds = [
      Index(U1Charge(np.random.randint(5, 15, D, dtype=np.int16)), False),
      Index(U1Charge(np.array([0, 1, 2, 3], dtype=np.int16)), False),
      Index(U1Charge(np.random.randint(5, 18, D, dtype=np.int16)), True)
  ]
  mpoinds = [
      Index(U1Charge(np.random.randint(0, 5, M)), False),
      Index(U1Charge(np.random.randint(0, 10, M)), True), mpsinds[1],
      mpsinds[1].flip_flow()
  ]
  Linds = [mpoinds[0].flip_flow(), mpsinds[0].flip_flow(), mpsinds[0]]
  Rinds = [mpoinds[1].flip_flow(), mpsinds[2].flip_flow(), mpsinds[2]]

  mps = BlockSparseTensor.random(mpsinds, dtype=dtype)
  mpo = BlockSparseTensor.random(mpoinds, dtype=dtype)
  L = BlockSparseTensor.random(Linds, dtype=dtype)
  R = BlockSparseTensor.random(Rinds, dtype=dtype)
  return L, mps, mpo, R


def get_tensor(R, num_charges, dtype=np.float64):
  Ds = np.random.randint(8, 12, R)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


def get_square_matrix(num_charges, dtype=np.float64):
  D = np.random.randint(40, 60)
  charges = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)

  flows = [False, True]
  indices = [Index(charges, flows[n]) for n in range(2)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


def get_hermitian_matrix(num_charges, dtype=np.float64):
  D = np.random.randint(40, 60)
  charges = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)

  flows = [False, True]
  indices = [Index(charges, flows[n]) for n in range(2)]
  A = BlockSparseTensor.random(indices=indices, dtype=dtype)
  return A + A.conj().T


def get_chargearray(num_charges, dtype=np.float64):
  D = np.random.randint(8, 12)
  charge = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  flow = False
  index = Index(charge, flow)
  return ChargeArray.random(indices=[index], dtype=dtype)


def get_contractable_tensors(R1, R2, cont, dtype, num_charges):
  DsA = np.random.randint(5, 10, R1)
  DsB = np.random.randint(5, 10, R2)
  assert R1 >= cont
  assert R2 >= cont
  chargesA = [
      BaseCharge(
          np.random.randint(-5, 6, (DsA[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R1 - cont)
  ]
  commoncharges = [
      BaseCharge(
          np.random.randint(-5, 6, (DsA[n + R1 - cont], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(cont)
  ]
  chargesB = [
      BaseCharge(
          np.random.randint(-5, 6, (DsB[n], num_charges)),
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
  for n, iA in enumerate(indsA):
    indicesA[iA] = Index(commoncharges[n], flowsA[iA])
    indicesB[indsB[n]] = Index(commoncharges[n], flowsB[indsB[n]])
  compA = list(set(np.arange(R1)) - set(indsA))
  compB = list(set(np.arange(R2)) - set(indsB))

  for n, cA in enumerate(compA):
    indicesA[cA] = Index(chargesA[n], flowsA[cA])
  for n, cB in enumerate(compB):
    indicesB[cB] = Index(chargesB[n], flowsB[cB])

  indices_final = []
  for n in sorted(compA):
    indices_final.append(indicesA[n])
  for n in sorted(compB):
    indices_final.append(indicesB[n])
  A = BlockSparseTensor.random(indices=indicesA, dtype=dtype)
  B = BlockSparseTensor.random(indices=indicesB, dtype=dtype)
  return A, B, indsA, indsB


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2, cont", [(4, 4, 2), (4, 3, 3), (3, 4, 3)])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_tensordot(R1, R2, cont, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a, b, indsa, indsb = get_contractable_tensors(R1, R2, cont, dtype,
                                                num_charges)
  actual = backend.tensordot(a, b, (indsa, indsb))
  expected = tensordot(a, b, (indsa, indsb))
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_reshape(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  shape = a.shape
  partitions = np.append(
      np.append(
          0,
          np.sort(
              np.random.choice(
                  np.arange(1, R), np.random.randint(1, R), replace=False))), R)
  new_shape = tuple([
      np.prod(shape[partitions[n - 1]:partitions[n]])
      for n in range(1, len(partitions))
  ])
  actual = backend.shape_tuple(backend.reshape(a, new_shape))
  assert actual == new_shape


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_transpose(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  order = np.arange(R)
  np.random.shuffle(order)
  actual = backend.transpose(a, order)
  expected = transpose(a, order)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_transpose_default(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  order = np.arange(R)[::-1]
  np.random.shuffle(order)
  actual = backend.transpose(a)
  expected = transpose(a, order)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


def test_shape_concat():
  backend = symmetric_backend.SymmetricBackend()
  a = np.asarray((2 * np.ones((1, 3, 1))))
  b = np.asarray(np.ones((1, 2, 1)))
  expected = backend.shape_concat((a, b), axis=1)
  actual = np.array([[[2.0], [2.0], [2.0], [1.0], [1.0]]])
  np.testing.assert_allclose(expected, actual)


def test_shape_tensor():

  backend = symmetric_backend.SymmetricBackend()
  a = np.asarray(np.ones([2, 3, 4]))
  assert isinstance(backend.shape_tensor(a), tuple)
  actual = backend.shape_tensor(a)
  expected = np.array([2, 3, 4])
  np.testing.assert_allclose(expected, actual)


def test_shape_tuple():
  backend = symmetric_backend.SymmetricBackend()
  a = np.asarray(np.ones([2, 3, 4]))
  actual = backend.shape_tuple(a)
  assert actual == (2, 3, 4)


def test_shape_prod():
  backend = symmetric_backend.SymmetricBackend()
  a = np.array(2 * np.ones([1, 2, 3, 4]))
  actual = np.array(backend.shape_prod(a))
  assert actual == 2**24


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_sqrt(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  actual = backend.sqrt(a)
  expected = sqrt(a)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (2, 3), (3, 3)])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_outer_product(R1, R2, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R1, num_charges, dtype)
  b = get_tensor(R2, num_charges, dtype)
  actual = backend.outer_product(a, b)
  expected = tensordot(a, b, 0)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_norm(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  assert backend.norm(a) == norm(a)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_eye(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  index = Index(
      BaseCharge(
          np.random.randint(-5, 6, (100, num_charges)),
          charge_types=[U1Charge] * num_charges), False)
  actual = backend.eye(index, dtype=dtype)
  expected = eye(index, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_eye_dtype(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  index = Index(
      BaseCharge(
          np.random.randint(-5, 6, (100, num_charges)),
          charge_types=[U1Charge] * num_charges), False)
  actual = backend.eye(index, dtype=dtype)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_ones(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.ones(indices, dtype=dtype)
  expected = ones(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_ones_dtype(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.ones(indices, dtype=dtype)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_zeros(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.zeros(indices, dtype=dtype)
  expected = zeros(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_zeros_dtype(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.zeros(indices, dtype=dtype)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_randn(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.randn(indices, dtype=dtype, seed=10)
  np.random.seed(10)
  expected = randn(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_randn_dtype(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.randn(indices, dtype=dtype, seed=10)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_random_uniform(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.random_uniform(indices, dtype=dtype, seed=10)
  np.random.seed(10)
  expected = random(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_random_uniform_dtype(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.random_uniform(indices, dtype=dtype, seed=10)
  assert actual.dtype == dtype


@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_randn_non_zero_imag(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.randn(indices, dtype=dtype, seed=10)
  assert np.linalg.norm(np.imag(actual.data)) != 0.0


@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_random_uniform_non_zero_imag(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  actual = backend.random_uniform(indices, dtype=dtype, seed=10)
  assert np.linalg.norm(np.imag(actual.data)) != 0.0


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_randn_seed(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  a = backend.randn(indices, dtype=dtype, seed=10)
  b = backend.randn(indices, dtype=dtype, seed=10)
  np.testing.assert_allclose(a.data, b.data)
  assert np.all([
      charge_equal(a._charges[n], b._charges[n])
      for n in range(len(a._charges))
  ])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_random_uniform_seed(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  a = backend.random_uniform(indices, dtype=dtype, seed=10)
  b = backend.random_uniform(indices, dtype=dtype, seed=10)
  np.testing.assert_allclose(a.data, b.data)
  assert np.all([
      charge_equal(a._charges[n], b._charges[n])
      for n in range(len(a._charges))
  ])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_random_uniform_boundaries(dtype, num_charges):
  np.random.seed(10)
  lb = 1.2
  ub = 4.8
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (10, num_charges)),
              charge_types=[U1Charge] * num_charges), False) for _ in range(R)
  ]
  a = backend.random_uniform(indices, seed=10, dtype=dtype)
  b = backend.random_uniform(indices, (lb, ub), seed=10, dtype=dtype)
  assert ((a.data >= 0).all() and (a.data <= 1).all() and
          (b.data >= lb).all() and (b.data <= ub).all())


@pytest.mark.parametrize(
    "dtype", [np.complex64, np.complex128, np.float64, np.float32, np.float16])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_conj(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  aconj = backend.conj(a)
  np.testing.assert_allclose(aconj.data, np.conj(a.data))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_addition(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  b = BlockSparseTensor.random(a.sparse_shape)
  res = backend.addition(a, b)
  np.testing.assert_allclose(res.data, a.data + b.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_addition_raises(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  b = get_tensor(R + 1, num_charges, dtype)
  with pytest.raises(ValueError):
    backend.addition(a, b)

  shape = b.sparse_shape
  c = BlockSparseTensor.random([shape[n] for n in reversed(range(len(shape)))])
  with pytest.raises(ValueError):
    backend.addition(a, c)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_subtraction(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  b = BlockSparseTensor.random(a.sparse_shape)
  res = backend.subtraction(a, b)

  np.testing.assert_allclose(res.data, a.data - b.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_subbtraction_raises(R, dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  b = get_tensor(R + 1, num_charges, dtype)
  with pytest.raises(ValueError):
    backend.subtraction(a, b)
  shape = b.sparse_shape
  c = BlockSparseTensor.random([shape[n] for n in reversed(range(len(shape)))])
  with pytest.raises(ValueError):
    backend.subtraction(a, c)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_multiply(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  res = backend.multiply(a, 5.1)
  np.testing.assert_allclose(res.data, a.data * 5.1)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_multiply_raises(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  with pytest.raises(TypeError):
    backend.multiply(a, np.array([5.1]))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_truediv(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  res = backend.divide(a, 5.1)
  np.testing.assert_allclose(res.data, a.data / 5.1)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_truediv_raises(dtype, num_charges):
  np.random.seed(10)
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, num_charges, dtype)
  with pytest.raises(TypeError):
    backend.divide(a, np.array([5.1]))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_eigh(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  H = get_hermitian_matrix(num_charges, dtype)
  eta, U = backend.eigh(H)
  eta_ac, U_ac = eigh(H)
  np.testing.assert_allclose(eta.data, eta_ac.data)
  np.testing.assert_allclose(U.data, U_ac.data)
  assert charge_equal(eta._charges[0], eta_ac._charges[0])
  assert np.all([
      charge_equal(U._charges[n], U_ac._charges[n])
      for n in range(len(U._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_matrix_inv(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  H = get_hermitian_matrix(num_charges, dtype)
  Hinv = backend.inv(H)
  Hinv_ac = inv(H)
  np.testing.assert_allclose(Hinv_ac.data, Hinv.data)
  assert np.all([
      charge_equal(Hinv._charges[n], Hinv_ac._charges[n])
      for n in range(len(Hinv._charges))
  ])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_matrix_inv_raises(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  H = get_tensor(3, num_charges, dtype)
  with pytest.raises(ValueError):
    backend.inv(H)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_broadcast_right_multiplication(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  Ds = [10, 30, 24]
  R = len(Ds)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), False) for n in range(R)
  ]
  tensor1 = backend.randn(indices, dtype=dtype)
  tensor2 = ChargeArray.random(
      indices=[indices[-1].copy().flip_flow()], dtype=dtype)
  t1dense = tensor1.todense()
  t2dense = tensor2.todense()
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  dense = t1dense * t2dense
  np.testing.assert_allclose(out.todense(), dense)


def test_broadcast_right_multiplication_raises():
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  num_charges = 1
  Ds = [10, 30, 24]
  R = len(Ds)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), False) for n in range(R)
  ]
  tensor1 = backend.randn(indices)
  tensor2 = ChargeArray.random(indices=indices)
  with pytest.raises(ValueError):
    backend.broadcast_right_multiplication(tensor1, tensor2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_broadcast_left_multiplication(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  Ds = [10, 30, 24]
  R = len(Ds)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), False) for n in range(R)
  ]

  tensor1 = ChargeArray.random(indices=[indices[0]], dtype=dtype)
  tensor2 = backend.randn(indices, dtype=dtype)
  t1dense = tensor1.todense()
  t2dense = tensor2.todense()
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  dense = np.reshape(t1dense, (10, 1, 1)) * t2dense
  np.testing.assert_allclose(out.todense(), dense)


def test_broadcast_left_multiplication_raises():
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  num_charges = 1
  Ds = [10, 30, 24]
  R = len(Ds)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), False) for n in range(R)
  ]

  tensor1 = ChargeArray.random(indices=indices)
  tensor2 = backend.randn(indices)
  with pytest.raises(ValueError):
    backend.broadcast_left_multiplication(tensor1, tensor2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_sparse_shape(dtype, num_charges):
  np.random.seed(10)
  Ds = [11, 12, 13]
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (Ds[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  a = BlockSparseTensor.random(indices=indices, dtype=dtype)
  backend = symmetric_backend.SymmetricBackend()
  for s1, s2 in zip(a.sparse_shape, backend.sparse_shape(a)):
    assert s1 == s2


#################################################################
# the following are sanity checks for eigsh_lanczos which do not
# really use block sparsity (all charges are identity charges)
#################################################################
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_valid_init_operator_with_shape_sanity_check(dtype):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  D = 16
  index = Index(U1Charge.random(D, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  a = BlockSparseTensor.random(indices, dtype=dtype)
  H = a + a.T.conj()

  def mv(vec, mat):
    return mat @ vec

  init = BlockSparseTensor.random([index], dtype=dtype)
  eta1, U1 = backend.eigsh_lanczos(mv, [H], init)
  v1 = np.reshape(U1[0].todense(), (D))
  v1 = v1 / sum(v1)

  eta2, U2 = np.linalg.eigh(H.todense())
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)

  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


def test_eigsh_small_number_krylov_vectors_sanity_check():
  np.random.seed(10)
  dtype = np.float64
  backend = symmetric_backend.SymmetricBackend()
  index = Index(U1Charge.random(2, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  H = BlockSparseTensor.random(indices, dtype=dtype)
  H.data = np.array([1, 2, 3, 4], dtype=np.float64)

  init = BlockSparseTensor.random([index], dtype=dtype)
  init.data = np.array([1, 1], dtype=np.float64)

  def mv(x, mat):
    return mat @ x

  eta, _ = backend.eigsh_lanczos(mv, [H], init, num_krylov_vecs=1)
  np.testing.assert_allclose(eta[0], 5)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_lanczos_sanity_check_1(dtype):
  np.random.seed(10)
  D = 16
  backend = symmetric_backend.SymmetricBackend()
  index = Index(U1Charge.random(D, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  H = BlockSparseTensor.random(indices, dtype=dtype)
  H = H + H.conj().T

  init = BlockSparseTensor.random([index], dtype=dtype)

  def mv(x, mat):
    return mat @ x

  eta1, U1 = backend.eigsh_lanczos(mv, [H], init)
  eta2, U2 = np.linalg.eigh(H.todense())
  v1 = np.reshape(U1[0].todense(), (D))
  v1 = v1 / sum(v1)

  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_lanczos_sanity_check_2(dtype):
  np.random.seed(10)
  D = 16
  backend = symmetric_backend.SymmetricBackend()
  index = Index(U1Charge.random(D, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  H = BlockSparseTensor.random(indices, dtype=dtype)
  H = H + H.conj().T

  def mv(x, mat):
    return mat @ x

  eta1, U1 = backend.eigsh_lanczos(
      mv, [H], shape=(H.sparse_shape[1].flip_flow(),), dtype=dtype)
  eta2, U2 = np.linalg.eigh(H.todense())
  v1 = np.reshape(U1[0].todense(), (D))
  v1 = v1 / sum(v1)

  v2 = U2[:, 0]
  v2 = v2 / sum(v2)

  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("numeig", [1, 2, 3, 4])
def test_eigsh_lanczos_reorthogonalize_sanity_check(dtype, numeig):
  np.random.seed(10)
  D = 24
  backend = symmetric_backend.SymmetricBackend()
  index = Index(U1Charge.random(D, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  H = BlockSparseTensor.random(indices, dtype=dtype)
  H = H + H.conj().T

  def mv(x, mat):
    return mat @ x

  eta1, U1 = backend.eigsh_lanczos(
      mv, [H],
      shape=(H.sparse_shape[1].flip_flow(),),
      dtype=dtype,
      numeig=numeig,
      num_krylov_vecs=D,
      reorthogonalize=True,
      ndiag=1,
      tol=10**(-12),
      delta=10**(-12))
  eta2, U2 = np.linalg.eigh(H.todense())

  np.testing.assert_allclose(eta1[0:numeig], eta2[0:numeig])
  for n in range(numeig):
    v2 = U2[:, n]
    v2 /= np.sum(v2)  #fix phases
    v1 = np.reshape(U1[n].todense(), (D))
    v1 /= np.sum(v1)

    np.testing.assert_allclose(v1, v2, rtol=10**(-5), atol=10**(-5))


#################################################################
# finished eigsh_lanczos sanity checks
#################################################################


################################################################
# non-trivial checks for eigsh_lanczos
################################################################
def finite_XXZ_mpo(Jz: np.ndarray, Jxy: np.ndarray, Bz: np.ndarray,
                   dtype=np.float64):
  N = len(Bz)
  mpo = []
  temp = np.zeros((1, 5, 2, 2), dtype=dtype)
  #BSz
  temp[0, 0, 0, 0] = -0.5 * Bz[0]
  temp[0, 0, 1, 1] = 0.5 * Bz[0]

  #Sm
  temp[0, 1, 0, 1] = Jxy[0] / 2.0 * 1.0
  #Sp
  temp[0, 2, 1, 0] = Jxy[0] / 2.0 * 1.0
  #Sz
  temp[0, 3, 0, 0] = Jz[0] * (-0.5)
  temp[0, 3, 1, 1] = Jz[0] * 0.5

  #11
  temp[0, 4, 0, 0] = 1.0
  temp[0, 4, 1, 1] = 1.0
  mpo.append(temp)
  for n in range(1, N - 1):
    temp = np.zeros((5, 5, 2, 2), dtype=dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #Sp
    temp[1, 0, 1, 0] = 1.0
    #Sm
    temp[2, 0, 0, 1] = 1.0
    #Sz
    temp[3, 0, 0, 0] = -0.5
    temp[3, 0, 1, 1] = 0.5
    #BSz
    temp[4, 0, 0, 0] = -0.5 * Bz[n]
    temp[4, 0, 1, 1] = 0.5 * Bz[n]

    #Sm
    temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
    #Sp
    temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
    #Sz
    temp[4, 3, 0, 0] = Jz[n] * (-0.5)
    temp[4, 3, 1, 1] = Jz[n] * 0.5
    #11
    temp[4, 4, 0, 0] = 1.0
    temp[4, 4, 1, 1] = 1.0

    mpo.append(temp)
  temp = np.zeros((5, 1, 2, 2), dtype=dtype)
  #11
  temp[0, 0, 0, 0] = 1.0
  temp[0, 0, 1, 1] = 1.0
  #Sp
  temp[1, 0, 1, 0] = 1.0
  #Sm
  temp[2, 0, 0, 1] = 1.0
  #Sz
  temp[3, 0, 0, 0] = -0.5
  temp[3, 0, 1, 1] = 0.5
  #BSz
  temp[4, 0, 0, 0] = -0.5 * Bz[-1]
  temp[4, 0, 1, 1] = 0.5 * Bz[-1]

  mpo.append(temp)
  return mpo


def blocksparse_XXZ_mpo(N, Jz=1, Jxy=1, Bz=0, dtype=np.float64):
  dense_mpo = finite_XXZ_mpo(
      Jz * np.ones(N - 1),
      Jxy * np.ones(N - 1),
      Bz=Bz * np.ones(N),
      dtype=dtype)
  ileft = Index(U1Charge(np.array([0])), False)
  iright = ileft.flip_flow()
  i1 = Index(U1Charge(np.array([0, -1, 1, 0, 0])), False)
  i2 = Index(U1Charge(np.array([0, -1, 1, 0, 0])), True)
  i3 = Index(U1Charge(np.array([0, 1])), False)
  i4 = Index(U1Charge(np.array([0, 1])), True)

  mpotensors = [BlockSparseTensor.fromdense(
      [ileft, i2, i3, i4], dense_mpo[0])] + [
          BlockSparseTensor.fromdense([i1, i2, i3, i4], tensor)
          for tensor in dense_mpo[1:-1]
      ] + [BlockSparseTensor.fromdense([i1, iright, i3, i4], dense_mpo[-1])]
  return mpotensors


def blocksparse_halffilled_spin_MPStensors(N=10, D=20, B=5, dtype=np.float64):
  auxcharges = [U1Charge([0])] + [
      U1Charge.random(D, n // 2, n // 2 + B) for n in range(N - 1)
  ] + [U1Charge([N // 2])]
  return [
      BlockSparseTensor.random([
          Index(auxcharges[n], False),
          Index(U1Charge([0, 1]), False),
          Index(auxcharges[n + 1], True)
      ], dtype=dtype) for n in range(N)
  ]


def blocksparse_DMRG_blocks(N=10,
                            D=20,
                            B=5,
                            Jz=1,
                            Jxy=1,
                            Bz=0,
                            dtype=np.float64):
  mps_tensors = blocksparse_halffilled_spin_MPStensors(N, D, B, dtype)
  mpo_tensors = blocksparse_XXZ_mpo(N, Jz, Jxy, Bz, dtype)
  mps = FiniteMPS(mps_tensors, backend='symmetric', canonicalize=True)
  mps.position(N // 2)
  mps_tensors = mps.tensors
  L = BlockSparseTensor.ones([
      mpo_tensors[0].sparse_shape[0].flip_flow(),
      mps_tensors[0].sparse_shape[0].flip_flow(), mps_tensors[0].sparse_shape[0]
  ])
  R = BlockSparseTensor.ones([
      mpo_tensors[-1].sparse_shape[1].flip_flow(),
      mps_tensors[-1].sparse_shape[2].flip_flow(),
      mps_tensors[-1].sparse_shape[2]
  ])
  for n in range(N // 2):
    L = ncon([L, mps_tensors[n], mps_tensors[n].conj(), mpo_tensors[n]],
             [[3, 1, 5], [1, 2, -2], [5, 4, -3], [3, -1, 4, 2]],
             backend='symmetric')
  for n in reversed(range(N // 2 + 1, N)):
    R = ncon([R, mps_tensors[n], mps_tensors[n].conj(), mpo_tensors[n]],
             [[3, 1, 5], [-2, 2, 1], [-3, 4, 5], [-1, 3, 4, 2]],
             backend='symmetric')
  return mps_tensors[N // 2], L, mpo_tensors[N // 2], R


@pytest.mark.parametrize('Jz', [1.0])
@pytest.mark.parametrize('Jxy', [1.0])
@pytest.mark.parametrize('Bz', [0.0, 0.2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('numeig, reorthogonalize', [(1, False), (4, True)])
def test_eigsh_lanczos_non_trivial(Jz, Jxy, Bz, dtype, numeig, reorthogonalize):
  N, D, B = 20, 100, 3

  def matvec(MPSTensor, LBlock, MPOTensor, RBlock, backend='symmetric'):
    return ncon([LBlock, MPSTensor, MPOTensor, RBlock],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend=backend)

  mps, L, mpo, R = blocksparse_DMRG_blocks(N, D, B, Jz, Jxy, Bz, dtype)
  backend = symmetric_backend.SymmetricBackend()
  np_backend = numpy_backend.NumPyBackend()

  eta_sym, U_sym = backend.eigsh_lanczos(
      matvec,
      args=[L, mpo, R, 'symmetric'],
      initial_state=mps,
      numeig=numeig,
      reorthogonalize=reorthogonalize,
      num_krylov_vecs=50)
  eta_np, U_np = np_backend.eigsh_lanczos(
      matvec,
      args=[L.todense(), mpo.todense(),
            R.todense(), 'numpy'],
      initial_state=mps.todense(),
      numeig=numeig,
      reorthogonalize=reorthogonalize,
      num_krylov_vecs=50)
  np.testing.assert_allclose(eta_sym, eta_np)
  for n, u in enumerate(U_sym):
    np.testing.assert_almost_equal(u.todense(), U_np[n])

################################################################
# non-trivial checks for eigsh_lanczos
################################################################

def test_eigsh_lanczos_raises():
  backend = symmetric_backend.SymmetricBackend()
  with pytest.raises(
      ValueError, match='`num_krylov_vecs` >= `numeig` required!'):
    backend.eigsh_lanczos(lambda x: x, numeig=10, num_krylov_vecs=9)
  with pytest.raises(
      ValueError,
      match="Got numeig = 2 > 1 and `reorthogonalize = False`. "
      "Use `reorthogonalize=True` for `numeig > 1`"):
    backend.eigsh_lanczos(lambda x: x, numeig=2, reorthogonalize=False)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, shape=(10,), dtype=None)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x, shape=None, dtype=np.float64)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigsh_lanczos(lambda x: x)
  with pytest.raises(
      TypeError, match="Expected a `BlockSparseTensor`. Got <class 'list'>"):
    backend.eigsh_lanczos(lambda x: x, initial_state=[1, 2, 3])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigsh_valid_init_operator_with_shape(dtype):
  np.random.seed(100)
  backend = symmetric_backend.SymmetricBackend()
  np_backend = numpy_backend.NumPyBackend()
  D = 16
  index = Index(U1Charge.random(D, -1, 1), True)
  indices = [index, index.copy().flip_flow()]

  a = BlockSparseTensor.random(indices, dtype=dtype)
  H = a + a.T.conj()

  def mv(vec, mat):
    return mat @ vec

  init = BlockSparseTensor.random([index], dtype=dtype)
  # note: this will only find eigenvalues in the charge (0,0)
  # block of H because `init` only has non-zero values there.
  # To find eigen values in other sectors we need to support non-zero
  # divergence for block-sparse tensors
  eta1, U1 = backend.eigsh_lanczos(mv, [H], init)
  eta2, U2 = np_backend.eigsh_lanczos(mv, [H.todense()], init.todense())

  v1 = np.reshape(U1[0].todense(), (D))
  v1 = v1 / sum(v1)
  v1 /= np.linalg.norm(v1)
  v2 = np.reshape(U2[0], (D))
  v2 = v2 / sum(v2)
  v2[np.abs(v2) < 1E-12] = 0.0
  v2 /= np.linalg.norm(v2)

  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_diagflat(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(3, num_charges, dtype)
  with pytest.raises(ValueError):
    backend.diagflat(a)
  b = get_chargearray(num_charges, dtype)
  expected = diag(b)
  actual = backend.diagflat(b)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])
  with pytest.raises(
      NotImplementedError, match="Can't specify k with Symmetric backend"):
    actual = backend.diagflat(b, k=1)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('Ds', [[200, 100], [100, 200]])
@pytest.mark.parametrize('flow', [False, True])
def test_diagonal(Ds, dtype, num_charges, flow):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
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
  diagonal = backend.diagonal(arr)

  sparse_blocks, _, block_shapes = _find_diagonal_sparse_blocks(
      arr.flat_charges, arr.flat_flows, 1)
  data = np.concatenate([
      np.diag(np.reshape(arr.data[sparse_blocks[n]], block_shapes[:, n]))
      for n in range(len(sparse_blocks))
  ])
  np.testing.assert_allclose(data, diagonal.data)
  np.testing.assert_allclose(unique_charges,
                             diagonal.flat_charges[0].unique_charges)
  with pytest.raises(NotImplementedError):
    diagonal = backend.diagonal(arr, axis1=0)
  with pytest.raises(NotImplementedError):
    diagonal = backend.diagonal(arr, axis2=1)
  with pytest.raises(NotImplementedError):
    diagonal = backend.diagonal(arr, offset=1)


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("axis1", [0, 1])
@pytest.mark.parametrize("axis2", [0, 1])
def test_trace(dtype, num_charges, offset, axis1, axis2):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  a = get_square_matrix(num_charges, dtype)
  if offset != 0:
    with pytest.raises(NotImplementedError):
      actual = backend.trace(a, offset=offset, axis1=axis1, axis2=axis2)
  else:
    if axis1 == axis2:
      with pytest.raises(ValueError):
        actual = backend.trace(a, offset=offset, axis1=axis1, axis2=axis2)
    else:
      actual = backend.trace(a, offset=offset, axis1=axis1, axis2=axis2)
      expected = trace(a, [axis1, axis2])
      np.testing.assert_allclose(actual.data, expected.data)

def test_pivot_not_implemented():
  backend = symmetric_backend.SymmetricBackend()
  with pytest.raises(NotImplementedError):
    backend.pivot(np.ones((2, 2)))


def test_eigsh_lanczos_caching():

  def matvec(mps, A, B, C):
    return ncon([A, mps, B, C],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend='symmetric')

  backend = symmetric_backend.SymmetricBackend()
  D = 100
  M = 5
  mpsinds = [
      Index(U1Charge(np.random.randint(5, 15, D, dtype=np.int16)), False),
      Index(U1Charge(np.array([0, 1, 2, 3], dtype=np.int16)), False),
      Index(U1Charge(np.random.randint(5, 18, D, dtype=np.int16)), True)
  ]
  mpoinds = [
      Index(U1Charge(np.random.randint(0, 5, M)), False),
      Index(U1Charge(np.random.randint(0, 10, M)), True), mpsinds[1],
      mpsinds[1].flip_flow()
  ]
  Linds = [mpoinds[0].flip_flow(), mpsinds[0].flip_flow(), mpsinds[0]]
  Rinds = [mpoinds[1].flip_flow(), mpsinds[2].flip_flow(), mpsinds[2]]
  mps = BlockSparseTensor.random(mpsinds)
  mpo = BlockSparseTensor.random(mpoinds)
  L = BlockSparseTensor.random(Linds)
  R = BlockSparseTensor.random(Rinds)
  ncv = 20
  backend.eigsh_lanczos(
      matvec, [L, mpo, R], initial_state=mps, num_krylov_vecs=ncv)
  assert get_cacher().cache == {}


def test_eigsh_lanczos_cache_exception():
  dtype = np.float64
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  D = 16
  index = Index(U1Charge.random(D, 0, 0), True)

  def mv(vec):
    raise ValueError()

  init = BlockSparseTensor.random([index], dtype=dtype)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(mv, [], init)
  cacher = get_cacher()
  assert not cacher.do_caching
  assert not get_caching_status()
  assert cacher.cache == {}


def compare_eigvals_and_eigvecs(U, eta, U_exact, eta_exact, thresh=1E-8):
  _, iy = np.nonzero(np.abs(eta[:, None] - eta_exact[None, :]) < thresh)
  U_exact_perm = U_exact[:, iy]
  U_exact_perm = U_exact_perm / np.expand_dims(np.sum(U_exact_perm, axis=0), 0)
  U = U / np.expand_dims(np.sum(U, axis=0), 0)
  np.testing.assert_allclose(U_exact_perm, U)
  np.testing.assert_allclose(eta, eta_exact[iy])


#################################################################
# the following is a sanity check for eigs which does not
# really use block sparsity (all charges are identity charges)
#################################################################
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigs_valid_init_operator_with_shape_sanity_check(dtype):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  D = 16
  index = Index(U1Charge.random(D, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  H = BlockSparseTensor.random(indices, dtype=dtype)

  def mv(vec, mat):
    return mat @ vec

  init = BlockSparseTensor.random([index], dtype=dtype)
  eta1, U1 = backend.eigs(mv, [H], init)

  eta2, U2 = np.linalg.eig(H.todense())

  compare_eigvals_and_eigvecs(
      np.stack([u.todense() for u in U1], axis=1), eta1, U2, eta2, thresh=1E-8)


def test_eigs_cache_exception():
  dtype = np.float64
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  D = 16
  index = Index(U1Charge.random(D, 0, 0), True)

  def mv(vec):
    raise ValueError()

  init = BlockSparseTensor.random([index], dtype=dtype)
  with pytest.raises(ValueError):
    backend.eigs(mv, [], init)
  cacher = get_cacher()
  assert not cacher.do_caching
  assert not get_caching_status()
  assert cacher.cache == {}


def test_eigs_raises():
  np.random.seed(10)
  dtype = np.float64
  backend = symmetric_backend.SymmetricBackend()
  D = 16
  index = Index(U1Charge.random(D, 0, 0), True)
  indices = [index, index.copy().flip_flow()]

  H = BlockSparseTensor.random(indices, dtype=dtype)
  init = BlockSparseTensor.random([index], dtype=dtype)

  with pytest.raises(
      ValueError, match='which = SI is currently not supported.'):
    backend.eigs(lambda x: x, [H], initial_state=init, which='SI')
  with pytest.raises(
      ValueError, match='which = LI is currently not supported.'):
    backend.eigs(lambda x: x, [H], initial_state=init, which='LI')
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigs(lambda x: x, [H])
  with pytest.raises(ValueError, match="`num_krylov_vecs`"):
    backend.eigs(lambda x: x, [H], numeig=3, num_krylov_vecs=3)
  with pytest.raises(TypeError, match="Expected a"):
    backend.eigs(lambda x: x, [H], initial_state=[])


#################################################################
# finished eigs sanity checks
#################################################################

################################################################
# non-trivial checks for eigs
################################################################

# TODO (martin): figure out why direct comparison of eigen vectors
# between tn.block_sparse.linalg.eig and eigs fails.

@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('numeig', [1, 4])
@pytest.mark.parametrize('x0', [True, False])
@pytest.mark.parametrize('args', [True, False])
def test_eigs_non_trivial(dtype, numeig, x0, args):
  L, mps, mpo, R = get_matvec_tensors(D=10, M=5, seed=10, dtype=dtype)
  def matvec(MPSTensor, LBlock, MPOTensor, RBlock):
    return ncon([LBlock, MPSTensor, MPOTensor, RBlock],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend='symmetric')

  def matvec_no_args(MPSTensor):
    return ncon([L, MPSTensor, mpo, R],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend='symmetric')

  backend = symmetric_backend.SymmetricBackend()
  if x0:
    init = mps
    _dtype = None
    shape = None
  else:
    init = None
    _dtype = dtype
    shape = mps.sparse_shape
  if args:
    mv = matvec
    _args = [L, mpo, R]
  else:
    mv = matvec_no_args
    _args = None

  eta_sym, U_sym = backend.eigs(
      A=mv,
      args=_args,
      initial_state=init,
      shape=shape,
      dtype=_dtype,
      numeig=numeig,
      num_krylov_vecs=50)

  H_sparse = ncon([L, mpo, R], [[1, -1, -4], [1, 2, -5, -2], [2, -3, -6]],
                  backend='symmetric')
  H_sparse.contiguous(inplace=True)
  D1, d, D2, _, _, _ = H_sparse.shape
  H_sparse = H_sparse.reshape((D1 * d * D2, D1 * d * D2))
  eta_sparse, _ = eig(H_sparse)
  mask = np.squeeze(eta_sparse.flat_charges[0].charges == 0)
  eigvals = eta_sparse.data[mask]
  isort = np.argsort(np.real(eigvals))[::-1]
  sparse_eigvals = eigvals[isort]
  _, iy = np.nonzero(np.abs(eta_sym[:, None] - sparse_eigvals[None, :]) < 1E-8)
  eta_exact = sparse_eigvals[iy]
  np.testing.assert_allclose(eta_exact, eta_sym)
  for n in range(numeig):
    assert norm(matvec(U_sym[n], L, mpo, R) - eta_sym[n] * U_sym[n]) < 1E-8

################################################################
# finished non-trivial checks for eigs
################################################################


def test_decomps_raise():
  np.random.seed(10)
  dtype = np.float64
  backend = symmetric_backend.SymmetricBackend()
  D = 16
  R = 3
  indices = [Index(U1Charge.random(D, -5, 5), True) for _ in range(R)]
  H = BlockSparseTensor.random(indices, dtype=dtype)
  with pytest.raises(
      NotImplementedError,
      match="Can't specify non_negative_diagonal with BlockSparse."):
    backend.qr(H, non_negative_diagonal=True)
  with pytest.raises(
      NotImplementedError,
      match="Can't specify non_negative_diagonal with BlockSparse."):
    backend.rq(H, non_negative_diagonal=True)


def test_convert_to_tensor_raises():
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  with pytest.raises(TypeError, match="cannot convert tensor of type"):
    backend.convert_to_tensor(np.random.rand(3, 3))


def test_einsum_raises():
  backend = symmetric_backend.SymmetricBackend()
  with pytest.raises(
      NotImplementedError, match="`einsum` currently not implemented"):
    backend.einsum('', [])

def test_sign():
  tensor = get_tensor(R=4, num_charges=1, dtype=np.float64)
  backend = symmetric_backend.SymmetricBackend()
  res = backend.sign(tensor)
  np.testing.assert_allclose(res.data, np.sign(tensor.data))

def test_abs():
  tensor = get_tensor(R=4, num_charges=1, dtype=np.float64)
  backend = symmetric_backend.SymmetricBackend()
  res = backend.abs(tensor)
  np.testing.assert_allclose(res.data, np.abs(tensor.data))


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('x0', [True, False])
@pytest.mark.parametrize('ncv', [None, 40])
def test_gmres(dtype, x0, ncv):
  backend = symmetric_backend.SymmetricBackend()
  L, mps, mpo, R = get_matvec_tensors(D=10, M=5, seed=10, dtype=dtype)
  b = randn_like(mps)

  def matvec(MPSTensor, LBlock, MPOTensor, RBlock):
    return ncon([LBlock, MPSTensor, MPOTensor, RBlock],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend='symmetric')

  if x0:
    init = mps
  else:
    init = None
  x, _ = backend.gmres(
      matvec,
      b, [L, mpo, R],
      x0=init,
      enable_caching=True,
      num_krylov_vectors=ncv)
  assert norm(matvec(x, L, mpo, R) - b) < 1E-10


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('x0', [True, False])
@pytest.mark.parametrize('ncv', [None, 40])
def test_gmres_no_args(dtype, x0, ncv):
  backend = symmetric_backend.SymmetricBackend()
  L, mps, mpo, R = get_matvec_tensors(D=10, M=5, seed=10, dtype=dtype)
  b = randn_like(mps)

  def matvec(MPSTensor):
    return ncon([L, MPSTensor, mpo, R],
                [[3, 1, -1], [1, 2, 4], [3, 5, -2, 2], [5, 4, -3]],
                backend='symmetric')

  if x0:
    init = mps
  else:
    init = None
  x, _ = backend.gmres(
      matvec,
      b,
      A_args=None,
      x0=init,
      enable_caching=True,
      num_krylov_vectors=ncv)
  assert norm(matvec(x) - b) < 1E-10


def test_gmres_cache_exception():
  backend = symmetric_backend.SymmetricBackend()
  _, mps, _, _ = get_matvec_tensors(D=10, M=5, seed=10, dtype=np.float64)
  b = randn_like(mps)

  def matvec(vec):
    raise ValueError()

  with pytest.raises(ValueError):
    backend.gmres(
        matvec,
        b,
        A_args=None,
        x0=mps,
        enable_caching=True,
        num_krylov_vectors=40)
  cacher = get_cacher()
  assert not cacher.do_caching
  assert not get_caching_status()
  assert cacher.cache == {}


def test_gmres_raises():
  backend = symmetric_backend.SymmetricBackend()
  _, mps, _, _ = get_matvec_tensors(D=10, M=5, seed=10, dtype=np.float64)

  with pytest.raises(ValueError, match="x0.sparse_shape"):
    b = randn_like(mps.conj())
    backend.gmres(lambda x: x, b, x0=mps)
  with pytest.raises(TypeError, match="x0.dtype"):
    b = BlockSparseTensor.random(mps.sparse_shape, dtype=np.complex128)
    backend.gmres(lambda x: x, b, x0=mps)
  b = randn_like(mps)
  with pytest.raises(ValueError, match="num_krylov_vectors must"):
    backend.gmres(lambda x: x, b, x0=mps, num_krylov_vectors=-1)
  with pytest.raises(ValueError, match="tol = "):
    backend.gmres(lambda x: x, b, x0=mps, tol=-0.001)
  with pytest.raises(ValueError, match="atol = "):
    backend.gmres(lambda x: x, b, x0=mps, atol=-0.001)

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("num_charges", [1, 2])
def test_item(dtype, num_charges):
  charges = BaseCharge(
      np.zeros((1, num_charges), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  indices = [Index(charges, True)]
  tensor = BlockSparseTensor.random(indices=indices, dtype=dtype)
  backend = symmetric_backend.SymmetricBackend()
  assert backend.item(tensor) == tensor.item()


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("num_charges", [1, 2])
def test_matmul(dtype, num_charges):
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  D = 100
  c1 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  c2 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  c3 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  charges1 = [c1, c2]
  charges2 = [c2, c3]
  flows1 = [False, True]
  flows2 = [False, True]
  inds1 = [Index(charges1[n], flows1[n]) for n in range(2)]
  inds2 = [Index(charges2[n], flows2[n]) for n in range(2)]
  A = BlockSparseTensor.random(indices=inds1, dtype=dtype)
  B = BlockSparseTensor.random(indices=inds2, dtype=dtype)

  actual = backend.matmul(A, B)
  expected = A @ B
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      charge_equal(expected._charges[n], actual._charges[n])
      for n in range(len(actual._charges))
  ])


def test_matmul_raises():
  dtype = np.float64
  num_charges = 1
  np.random.seed(10)
  backend = symmetric_backend.SymmetricBackend()
  D = 100
  c1 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  c2 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  c3 = BaseCharge(
      np.random.randint(-5, 6, (D, num_charges)),
      charge_types=[U1Charge] * num_charges)
  charges1 = [c1, c2, c3]
  charges2 = [c2, c3]
  flows1 = [False, True, False]
  flows2 = [False, True]
  inds1 = [Index(charges1[n], flows1[n]) for n in range(3)]
  inds2 = [Index(charges2[n], flows2[n]) for n in range(2)]
  A = BlockSparseTensor.random(indices=inds1, dtype=dtype)
  B = BlockSparseTensor.random(indices=inds2, dtype=dtype)
  with pytest.raises(ValueError, match="inputs to"):
    _ = backend.matmul(A, B)

@pytest.mark.parametrize("dtype", np_dtypes)
def test_eps(dtype):
  backend = symmetric_backend.SymmetricBackend()
  assert backend.eps(dtype) == np.finfo(dtype).eps
