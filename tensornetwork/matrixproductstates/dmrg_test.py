from tensornetwork.matrixproductstates import FiniteMPS
from tensornetwork.matrixproductstates import FiniteDMRG
from tensornetwork.matrixproductstates.mpo import FiniteXXZ


def test_BaseDMRG_init(backend):
  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  dtype = np.float64
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype)
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype)
  dmrg = BaseDMRG(mps, mpo, np.ones((1, 1, 1), dtype=dtype),
                  np.ones((1, 1, 1), dtype=dtype), 'name')
  assert dmrg.name == 'name'
  assert dmrg.backend.name == backend
  assert dmrg.dtype == dtype


def test_finite_DMRG_init(backend):
  tn.set_default_backend(backend)
  N = 10
  D = 10
  Jz = np.ones(N - 1)
  Jxy = np.ones(N - 1)
  Bz = np.zeros(N)
  dtype = np.float64
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype)
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype)
  dmrg = FiniteDMRG(mps, mpo)
