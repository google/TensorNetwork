from tensornetwork import FiniteMPS
from tensornetwork.matrixproductstates.dmrg import FiniteDMRG, BaseDMRG
from tensornetwork.backends import backend_factory
from tensornetwork.matrixproductstates.mpo import FiniteXXZ
import pytest
import numpy as np


@pytest.fixture(
    name="backend_dtype_values",
    params=[('numpy', np.float64), ('numpy', np.complex128),
            ('jax', np.float64), ('jax', np.complex128),
            ('pytorch', np.float64)])
def backend_dtype(request):
  return request.param


def get_XXZ_Hamiltonian(N, Jx, Jy, Jz):
  Sx = {}
  Sy = {}
  Sz = {}
  sx = np.array([[0, 0.5], [0.5, 0]])
  sy = np.array([[0, 0.5], [-0.5, 0]])
  sz = np.diag([-0.5, 0.5])
  for n in range(N):
    Sx[n] = np.kron(np.kron(np.eye(2**n), sx), np.eye(2**(N - 1 - n)))
    Sy[n] = np.kron(np.kron(np.eye(2**n), sy), np.eye(2**(N - 1 - n)))
    Sz[n] = np.kron(np.kron(np.eye(2**n), sz), np.eye(2**(N - 1 - n)))
  H = np.zeros((2**N, 2**N))
  for n in range(N - 1):
    H += Jx * Sx[n] @ Sx[n + 1] - Jy * Sy[n] @ Sy[n + 1] + Jz * Sz[n] @ Sz[n +
                                                                           1]
  return H


def test_BaseDMRG_init(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=backend)
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = BaseDMRG(mps, mpo, np.ones((1, 1, 1), dtype=dtype),
                  np.ones((1, 1, 1), dtype=dtype), 'name')
  assert dmrg.name == 'name'
  assert dmrg.backend is backend


def test_BaseDMRG_raises():
  numpy_backend = backend_factory.get_backend('numpy')
  pytorch_backend = backend_factory.get_backend('pytorch')
  dtype = np.float64
  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=numpy_backend)
  mps = FiniteMPS.random(
      [2] * (N - 1), [D] * (N - 2), dtype=dtype, backend=numpy_backend)
  with pytest.raises(ValueError):
    BaseDMRG(mps, mpo, numpy_backend.ones((1, 1, 1), dtype=dtype),
             numpy_backend.ones((1, 1, 1), dtype=dtype), 'name')

  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=np.float64, backend=numpy_backend)
  mps = FiniteMPS.random(
      [2] * N, [D] * (N - 1), dtype=np.float32, backend=numpy_backend)
  with pytest.raises(
      TypeError,
      match="mps.dtype = {} is different from "
      "mpo.dtype = {}".format(mps.dtype, mpo.dtype)):
    BaseDMRG(mps, mpo, numpy_backend.ones((1, 1, 1), dtype=dtype),
             numpy_backend.ones((1, 1, 1), dtype=dtype), 'name')

  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=np.float64, backend=numpy_backend)
  mps = FiniteMPS.random(
      [2] * N, [D] * (N - 1), dtype=np.float64, backend=pytorch_backend)
  with pytest.raises(TypeError, match="mps and mpo use different backends."):
    BaseDMRG(mps, mpo, numpy_backend.ones((1, 1, 1), dtype=dtype),
             numpy_backend.ones((1, 1, 1), dtype=dtype), 'name')


def test_BaseDMRG_raises_2():
  backend = 'numpy'
  backend_obj = backend_factory.get_backend(backend)
  dtype = np.float64

  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=backend)
  tensors = [np.random.randn(1, 2, D)] + [
      np.random.randn(D, 2, D) for _ in range(N - 2)
  ] + [np.random.randn(D, 2, 1)]
  mps = FiniteMPS(
      tensors, center_position=None, canonicalize=False, backend=backend)
  with pytest.raises(
      ValueError,
      match="Found mps in non-canonical form. Please canonicalize mps."):
    BaseDMRG(mps, mpo, backend_obj.ones((1, 1, 1), dtype=dtype),
             backend_obj.ones((1, 1, 1), dtype=dtype), 'name')


def test_BaseDMRG_position(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=backend)
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = BaseDMRG(mps, mpo, np.ones((1, 1, 1), dtype=dtype),
                  np.ones((1, 1, 1), dtype=dtype), 'name')
  dmrg.position(N - 1)
  np.testing.assert_allclose(np.arange(N), sorted(list(dmrg.left_envs.keys())))
  np.testing.assert_allclose([N - 1], list(dmrg.right_envs.keys()))
  assert dmrg.mps.center_position == N - 1
  dmrg.position(0)
  np.testing.assert_allclose([0], list(dmrg.left_envs.keys()))
  np.testing.assert_allclose(np.arange(N), sorted(list(dmrg.right_envs.keys())))
  assert dmrg.mps.center_position == 0

  with pytest.raises(IndexError, match="site > length of mps"):
    dmrg.position(N)
  with pytest.raises(IndexError, match="site < 0"):
    dmrg.position(-1)


def test_compute_envs(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=backend)
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = BaseDMRG(mps, mpo, np.ones((1, 1, 1), dtype=dtype),
                  np.ones((1, 1, 1), dtype=dtype), 'name')
  dmrg.position(5)
  dmrg.compute_left_envs()
  dmrg.compute_right_envs()
  np.testing.assert_allclose([0, 1, 2, 3, 4, 5],
                             sorted(list(dmrg.left_envs.keys())))
  np.testing.assert_allclose([5, 6, 7, 8, 9],
                             sorted(list(dmrg.right_envs.keys())))


@pytest.mark.parametrize("N", [4, 6, 7])
def test_finite_DMRG_init(backend_dtype_values, N):
  np.random.seed(16)
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  H = get_XXZ_Hamiltonian(N, 1, 1, 1)
  eta, _ = np.linalg.eigh(H)

  mpo = FiniteXXZ(
      Jz=np.ones(N - 1),
      Jxy=np.ones(N - 1),
      Bz=np.zeros(N),
      dtype=dtype,
      backend=backend)
  D = 32
  # test one-site DMRG
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = FiniteDMRG(mps, mpo)
  one_site_energy = dmrg.run_one_site(num_sweeps=4, num_krylov_vecs=10)
  np.testing.assert_allclose(one_site_energy, eta[0])
  one_site_energy_no_sweeps = dmrg.run_one_site(num_sweeps=0,
                                                num_krylov_vecs=10)
  np.testing.assert_allclose(one_site_energy_no_sweeps, one_site_energy)
  # test two-site DMRG
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = FiniteDMRG(mps, mpo)
  two_site_energy = dmrg.run_two_site(max_bond_dim=D, num_sweeps=4,
                                      num_krylov_vecs=10)
  np.testing.assert_allclose(two_site_energy, eta[0])
  two_site_energy_no_sweeps = dmrg.run_two_site(max_bond_dim=D, num_sweeps=0,
                                                num_krylov_vecs=10)
  np.testing.assert_allclose(two_site_energy_no_sweeps, two_site_energy)


def test_finite_DMRG_one_site_outstream(backend_dtype_values, capsys):
  np.random.seed(16)
  N = 6
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  mpo = FiniteXXZ(
      Jz=np.ones(N - 1),
      Jxy=np.ones(N - 1),
      Bz=np.zeros(N),
      dtype=dtype,
      backend=backend)
  D = 32
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = FiniteDMRG(mps, mpo)
  num_sweeps = 2
  dmrg.run_one_site(
      num_sweeps=num_sweeps, num_krylov_vecs=10, verbose=2, precision=1E-100)
  out, _ = capsys.readouterr()
  out = out.split('\n')
  act = [o[:28] + '\n' for o in out]
  act = ''.join(act[0:num_sweeps * (2 * N - 2)])

  exp = ''.join([
      f"SS-DMRG sweep={n}/{num_sweeps}, site={m}/{N}:\n"
      for n in range(1, num_sweeps + 1)
      for m in [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]
  ])
  assert act == exp

def test_finite_DMRG_two_site_outstream(backend_dtype_values, capsys):
  np.random.seed(16)
  N = 6
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  mpo = FiniteXXZ(
      Jz=np.ones(N - 1),
      Jxy=np.ones(N - 1),
      Bz=np.zeros(N),
      dtype=dtype,
      backend=backend)
  D = 32
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = FiniteDMRG(mps, mpo)
  num_sweeps = 2
  dmrg.run_two_site(max_bond_dim=D, num_sweeps=num_sweeps, num_krylov_vecs=10,
                    verbose=2, precision=1E-100)
  out, _ = capsys.readouterr()
  out = out.split('\n')
  act = [o[:33] + '\n' for o in out]
  act = ''.join(act[0:num_sweeps * (2 * N - 2)])

  exp = ''.join([
      f"TS-DMRG sweep={n}/{num_sweeps}, sites=({left_site},{left_site + 1})"
      f"/{N}:\n"
      for n in range(1, num_sweeps + 1)
      for left_site in [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
  ])
  assert act == exp
