import tensornetwork as tn
import numpy as np
from tensornetwork import (FiniteMPO, FiniteMPS, U1Charge, Index,
                           BlockSparseTensor, FiniteXXZ, FiniteDMRG)


def blocksparse_XXZ_mpo(Jz, Jxy, Bz, dtype=np.float64):
  dense_mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype).tensors
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
  return FiniteMPO(mpotensors, backend='symmetric')


def blocksparse_halffilled_spin_mps(N=10, D=20, B=5, dtype=np.float64):
  auxcharges = [U1Charge([0])] + [
      U1Charge.random(D, n // 2, n // 2 + B) for n in range(N - 1)
  ] + [U1Charge([N // 2])]
  tensors = [
      BlockSparseTensor.random([
          Index(auxcharges[n], False),
          Index(U1Charge([0, 1]), False),
          Index(auxcharges[n + 1], True)
      ],
                               dtype=dtype) for n in range(N)
  ]
  return FiniteMPS(tensors, canonicalize=True, backend='symmetric')


def initialize_spin_mps(N, D, dtype, backend):
  if backend == 'symmetric':
    return blocksparse_halffilled_spin_mps(N=N, D=D, B=5, dtype=dtype)
  return FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)


def initialize_XXZ_mpo(Jz, Jxy, Bz, dtype, backend):
  if backend == 'symmetric':
    return blocksparse_XXZ_mpo(Jz=Jz, Jxy=Jxy, Bz=Bz, dtype=dtype)
  return FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=backend)


def run_twosite_dmrg(N, D, dtype, Jz, Jxy, Bz, num_sweeps, backend):
  mps = initialize_spin_mps(N, D, dtype, backend)
  mpo = initialize_XXZ_mpo(Jz, Jxy, Bz, dtype, backend)
  dmrg = FiniteDMRG(mps, mpo)
  return dmrg.run_two_site(
      max_bond_dim=100, num_sweeps=num_sweeps, num_krylov_vecs=10)


num_sites, bond_dim, datatype = 10, 20, np.float64
jz = np.ones(num_sites - 1)
jxy = np.ones(num_sites - 1)
bz = np.zeros(num_sites)
n_sweeps = 2
energies = {}
backends = ('jax', 'numpy', 'symmetric', 'pytorch')
for be in backends:
  print(f'\nrunning DMRG for {be} backend')
  energies[be] = run_twosite_dmrg(
      num_sites,
      bond_dim,
      datatype,
      jz,
      jxy,
      bz,
      num_sweeps=n_sweeps,
      backend=be)

text = [
    f"\nenergy for backend {backend}: {e}" for backend, e in energies.items()
]
print(''.join(text))
