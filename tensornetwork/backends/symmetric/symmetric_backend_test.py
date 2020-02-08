"""Tests for graphmode_tensornetwork."""
import tensorflow as tf
import numpy as np
import pytest
from tensornetwork.backends.symmetric import symmetric_backend
from tensornetwork.block_tensor.charge import U1Charge
from tensornetwork.block_tensor.index import Index
from tensornetwork.block_tensor.block_tensor import tensordot, BlockSparseTensor, transpose, sqrt, ChargeArray, diag, trace, norm, eye, ones, zeros, randn, rand, eigh, inv

np_randn_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_randn_dtypes + [np.complex64, np.complex128]
np_tensordot_dtypes = [np.float16, np.float64, np.complex128]


def get_tensor(R, dtype=np.float64):
  Ds = np.random.randint(8, 12, R)
  charges = [U1Charge(np.random.randint(-5, 5, Ds[n])) for n in range(R)]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


def get_square_matrix(dtype=np.float64):
  D = np.random.randint(40, 60)
  charges = U1Charge(np.random.randint(-5, 5, D))
  flows = [False, True]
  indices = [Index(charges, flows[n]) for n in range(2)]
  return BlockSparseTensor.random(indices=indices, dtype=dtype)


def get_hermitian_matrix(dtype=np.float64):
  D = np.random.randint(40, 60)
  charges = U1Charge(np.random.randint(-5, 5, D))
  flows = [False, True]
  indices = [Index(charges, flows[n]) for n in range(2)]
  A = BlockSparseTensor.random(indices=indices, dtype=dtype)
  return A + A.conj().T


def get_chargearray(dtype=np.float64):
  D = np.random.randint(8, 12)
  charge = U1Charge(np.random.randint(-5, 5, D))
  flow = False
  index = Index(charge, flow)
  return ChargeArray.random(index=index, dtype=dtype)


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
  for n in range(len(indsA)):
    indicesA[indsA[n]] = Index(commoncharges[n], flowsA[indsA[n]])
    indicesB[indsB[n]] = Index(commoncharges[n], flowsB[indsB[n]])
  compA = list(set(np.arange(R1)) - set(indsA))
  compB = list(set(np.arange(R2)) - set(indsB))

  for n in range(len(compA)):
    indicesA[compA[n]] = Index(chargesA[n], flowsA[compA[n]])
  for n in range(len(compB)):
    indicesB[compB[n]] = Index(chargesB[n], flowsB[compB[n]])
  indices_final = []
  for n in sorted(compA):
    indices_final.append(indicesA[n])
  for n in sorted(compB):
    indices_final.append(indicesB[n])
  shapes = tuple([i.dim for i in indices_final])
  A = BlockSparseTensor.random(indices=indicesA, dtype=dtype)
  B = BlockSparseTensor.random(indices=indicesB, dtype=dtype)
  return A, B, indsA, indsB


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2, cont", [(4, 4, 2), (4, 3, 3), (3, 4, 3)])
def test_tensordot(R1, R2, cont, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a, b, indsa, indsb = get_contractable_tensors(R1, R2, cont, dtype)
  actual = backend.tensordot(a, b, (indsa, indsb))
  expected = tensordot(a, b, (indsa, indsb))
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5, 6, 7])
def test_reshape(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
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
def test_transpose(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  order = np.arange(R)
  np.random.shuffle(order)
  actual = backend.transpose(a, order)
  expected = transpose(a, order)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
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
def test_sqrt(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  actual = backend.sqrt(a)
  expected = sqrt(a)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
def test_diag(dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(3, dtype)
  with pytest.raises(TypeError):
    assert backend.diag(a)
  b = get_chargearray(dtype)
  actual = backend.diag(b)
  expected = diag(b)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
def test_trace(dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_square_matrix(dtype)
  actual = backend.trace(a)
  expected = trace(a)
  np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (2, 3), (3, 3)])
def test_outer_product(R1, R2, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R1, dtype)
  b = get_tensor(R2, dtype)
  actual = backend.outer_product(a, b)
  expected = tensordot(a, b, 0)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_norm(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  assert backend.norm(a) == norm(a)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye(dtype):
  backend = symmetric_backend.SymmetricBackend()
  index = Index(U1Charge.random(-5, 5, 100), False)
  actual = backend.eye(index, dtype=dtype)
  expected = eye(index, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_dtypes)
def test_eye_dtype(dtype):
  backend = symmetric_backend.SymmetricBackend()
  index = Index(U1Charge.random(-5, 5, 100), False)
  actual = backend.eye(index, dtype=dtype)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_ones(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.ones(indices, dtype=dtype)
  expected = ones(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_ones_dtype(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.ones(indices, dtype=dtype)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_zeros(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.zeros(indices, dtype=dtype)
  expected = zeros(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_zeros_dtype(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.zeros(indices, dtype=dtype)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_randn(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.randn(indices, dtype=dtype, seed=10)
  np.random.seed(10)
  expected = randn(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_dtype(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.randn(indices, dtype=dtype, seed=10)
  assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", np_randn_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_random_uniform(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.random_uniform(indices, dtype=dtype, seed=10)
  np.random.seed(10)
  expected = rand(indices, dtype=dtype)
  np.testing.assert_allclose(expected.data, actual.data)
  assert np.all([
      expected.indices[n] == actual.indices[n]
      for n in range(len(actual.indices))
  ])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_dtype(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.random_uniform(indices, dtype=dtype, seed=10)
  assert actual.dtype == dtype


@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_randn_non_zero_imag(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.randn(indices, dtype=dtype, seed=10)
  assert np.linalg.norm(np.imag(actual.data)) != 0.0


@pytest.mark.parametrize("R", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_random_uniform_non_zero_imag(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  actual = backend.random_uniform(indices, dtype=dtype, seed=10)
  assert np.linalg.norm(np.imag(actual.data)) != 0.0


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_randn_seed(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  a = backend.randn(indices, dtype=dtype, seed=10)
  b = backend.randn(indices, dtype=dtype, seed=10)
  np.testing.assert_allclose(a.data, b.data)
  assert np.all([a.indices[n] == b.indices[n] for n in range(len(a.indices))])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_seed(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  a = backend.random_uniform(indices, dtype=dtype, seed=10)
  b = backend.random_uniform(indices, dtype=dtype, seed=10)
  np.testing.assert_allclose(a.data, b.data)
  assert np.all([a.indices[n] == b.indices[n] for n in range(len(a.indices))])


@pytest.mark.parametrize("dtype", np_randn_dtypes)
def test_random_uniform_boundaries(dtype):
  lb = 1.2
  ub = 4.8
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(R)]
  a = backend.random_uniform(indices, seed=10, dtype=dtype)
  b = backend.random_uniform(indices, (lb, ub), seed=10, dtype=dtype)
  assert ((a.data >= 0).all() and (a.data <= 1).all() and
          (b.data >= lb).all() and (b.data <= ub).all())


@pytest.mark.parametrize(
    "dtype", [np.complex64, np.complex128, np.float64, np.float32, np.float16])
def test_conj(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  aconj = backend.conj(a)
  np.testing.assert_allclose(aconj.data, np.conj(a.data))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_addition(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  b = BlockSparseTensor.random(a.indices)
  res = a + b
  np.testing.assert_allclose(res.data, a.data + b.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_addition_raises(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  b = get_tensor(R + 1, dtype)
  with pytest.raises(ValueError):
    res = a + b
  c = BlockSparseTensor.random(
      [a.indices[n] for n in reversed(range(len(a.indices)))])
  with pytest.raises(ValueError):
    res = a + c


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_subtraction(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  b = BlockSparseTensor.random(a.indices)
  res = a - b
  np.testing.assert_allclose(res.data, a.data - b.data)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_subbtraction_raises(R, dtype):
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  b = get_tensor(R + 1, dtype)
  with pytest.raises(ValueError):
    res = a - b

  c = BlockSparseTensor.random(
      [a.indices[n] for n in reversed(range(len(a.indices)))])
  with pytest.raises(ValueError):
    res = a - c


@pytest.mark.parametrize("dtype", np_dtypes)
def test_multiply(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  res = a * 5.1
  np.testing.assert_allclose(res.data, a.data * 5.1)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_multiply_raises(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  with pytest.raises(TypeError):
    res = a * np.array([5.1])


@pytest.mark.parametrize("dtype", np_dtypes)
def test_truediv(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  res = a / 5.1
  np.testing.assert_allclose(res.data, a.data / 5.1)


@pytest.mark.parametrize("dtype", np_dtypes)
def test_truediv_raises(dtype):
  R = 4
  backend = symmetric_backend.SymmetricBackend()
  a = get_tensor(R, dtype)
  with pytest.raises(TypeError):
    res = a / np.array([5.1])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_eigh(dtype):

  backend = symmetric_backend.SymmetricBackend()
  H = get_hermitian_matrix(dtype)
  eta, U = backend.eigh(H)
  eta_ac, U_ac = eigh(H)
  np.testing.assert_allclose(eta.data, eta_ac.data)
  np.testing.assert_allclose(U.data, U_ac.data)
  assert eta.index == eta_ac.index
  assert np.all(
      [U.indices[n] == U_ac.indices[n] for n in range(len(U.indices))])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_matrix_inv(dtype):
  backend = symmetric_backend.SymmetricBackend()
  H = get_hermitian_matrix(dtype)
  Hinv = backend.inv(H)
  Hinv_ac = inv(H)
  np.testing.assert_allclose(Hinv_ac.data, Hinv.data)
  assert np.all(
      [Hinv.indices[n] == Hinv_ac.indices[n] for n in range(len(Hinv.indices))])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_matrix_inv_raises(dtype):
  backend = symmetric_backend.SymmetricBackend()
  H = get_tensor(3, dtype)
  with pytest.raises(ValueError):
    Hinv = backend.inv(H)
