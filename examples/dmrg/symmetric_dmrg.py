"""
Example of backend independent DMRG calculation for the
spin 1/2 Heisenberg Model.
"""

from typing import Type, Text
import tensornetwork as tn
import numpy as np
import tensornetwork as tn

import jax
#enable double precision in JAX
jax.config.update('jax_enable_x64', True)


def blocksparse_XXZ_mpo(Jz: np.ndarray,
                        Jxy: np.ndarray,
                        Bz: np.ndarray,
                        dtype: Type[np.number] = np.float64) -> tn.FiniteMPO:
  """
  Prepare a symmetric MPO.

  Args:
    Jz, Jxy, Bz: Hamiltonian parameters.
    dtype: data type.

  Returns:
    `tn.FiniteMPO`: The mpo of the XXZ Heisenberg model with U(1) symmetry.
  """
  dense_mpo = tn.FiniteXXZ(Jz, Jxy, Bz, dtype=dtype).tensors
  ileft = tn.Index(tn.U1Charge(np.array([0])), False)
  iright = ileft.flip_flow()
  i1 = tn.Index(tn.U1Charge(np.array([0, -1, 1, 0, 0])), False)
  i2 = tn.Index(tn.U1Charge(np.array([0, -1, 1, 0, 0])), True)
  i3 = tn.Index(tn.U1Charge(np.array([0, 1])), False)
  i4 = tn.Index(tn.U1Charge(np.array([0, 1])), True)

  mpotensors = [tn.BlockSparseTensor.fromdense(
      [ileft, i2, i3, i4], dense_mpo[0])] + [
          tn.BlockSparseTensor.fromdense([i1, i2, i3, i4], tensor)
          for tensor in dense_mpo[1:-1]
      ] + [tn.BlockSparseTensor.fromdense([i1, iright, i3, i4], dense_mpo[-1])]
  return tn.FiniteMPO(mpotensors, backend='symmetric')


def blocksparse_halffilled_spin_mps(N: int,
                                    D: int,
                                    B: int = 5,
                                    dtype: Type[np.number] = np.float64):
  """
  Prepare a U(1) symmetric spin 1/2 MPS at zero total magnetization.

  Args:
    N: Number of spins.
    D: The bond dimension.
    B: The number of symmetry sectors on each ancillary link.
    dtype: The data type of the MPS.

  Returns:
    `tn.FiniteMPS`: A U(1) symmetric spin 1/2 mps at zero total magnetization.
  """
  auxcharges = [tn.U1Charge([0])] + [
      tn.U1Charge.random(D, n // 2, n // 2 + B) for n in range(N - 1)
  ] + [tn.U1Charge([N // 2])]
  tensors = [
      tn.BlockSparseTensor.random([
          tn.Index(auxcharges[n], False),
          tn.Index(tn.U1Charge([0, 1]), False),
          tn.Index(auxcharges[n + 1], True)
      ],
                               dtype=dtype) for n in range(N)
  ]
  return tn.FiniteMPS(tensors, canonicalize=True, backend='symmetric')


def initialize_spin_mps(N: int, D: int, dtype: Type[np.number], backend: Text):
  """
  Helper function to initialize an MPS for a given backend.

  Args:
    N: Number of spins.
    D: The bond dimension.
    dtype: The data type of the MPS.

  Returns:
    `tn.FiniteMPS`: A spin 1/2 mps for the corresponding backend.
  """
  if backend == 'symmetric':
    return blocksparse_halffilled_spin_mps(N=N, D=D, B=5, dtype=dtype)
  return tn.FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)


def initialize_XXZ_mpo(Jz: np.ndarray, Jxy: np.ndarray, Bz: np.ndarray,
                       dtype: Type[np.number], backend: Text):
  """
  Helper function to initialize the XXZ Heisenberg MPO
  for a given backend.

  Args:
    Jz, Jxy, Bz: Hamiltonian parameters.
    dtype: data type.
    backend: The backend.
  Returns:
    `tn.FiniteMPS`: A spin 1/2 mps for the corresponding backend.
  """

  if backend == 'symmetric':
    return blocksparse_XXZ_mpo(Jz=Jz, Jxy=Jxy, Bz=Bz, dtype=dtype)
  return tn.FiniteXXZ(Jz, Jxy, Bz, dtype=dtype, backend=backend)


def run_twosite_dmrg(N: int, D: int, dtype: Type[np.number], Jz: np.ndarray,
                     Jxy: np.ndarray, Bz: np.ndarray, num_sweeps: int,
                     backend: Text):
  """
  Run two-site dmrg for the XXZ Heisenberg model using a given backend.

  Args:
    N: Number of spins.
    D: The bond dimension.
    dtype: The data type of the MPS.
    Jz, Jxy, Bz: Hamiltonian parameters.
    num_sweeps: Number of DMRG sweeps to perform.
    backend: The backend.

  Returns:
    float/complex: The energy upon termination of DMRG.

  """
  mps = initialize_spin_mps(N, 32, dtype, backend)
  mpo = initialize_XXZ_mpo(Jz, Jxy, Bz, dtype, backend)
  dmrg = tn.FiniteDMRG(mps, mpo)
  return dmrg.run_two_site(
      max_bond_dim=D, num_sweeps=num_sweeps, num_krylov_vecs=10, verbose=1)


if __name__ == '__main__':
  # Run two-site DMRG for the XXZ Heisenberg model for
  # different backends.
  #
  # change parameters to simulate larger systems and bigger
  # bond dimensions
  #
  # Notes: JAX backend peforms jit (just in time) compilation of
  #        operations. This results in an overhead whenever the computation
  #        encounters a bond dimension it has not seen before.
  #        In two-site DMRG this happens when the bond dimensions
  #        are ramped up during the simulation close to the boundaries.
  #
  #        The symmetric backend is for small bond dimensions typicall slower
  #        than other backends, due to inherent book-keeping overhead.
  #        In comparison with numpy, the two backends typically are of the same
  #        speed for a bond dimension of D ~ 100. For value of D >~ 400, the
  #        symmetric backend is typically substantially faster than numpy,
  #        pytorch or jax on CPU.

  num_sites, bond_dim, datatype = 20, 16, np.float64
  jz = np.ones(num_sites - 1)
  jxy = np.ones(num_sites - 1)
  bz = np.zeros(num_sites)
  n_sweeps = 5
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
