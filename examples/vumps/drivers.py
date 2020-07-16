"""
Interface functions for VUMPS.
"""
import sys
import os
import numpy as np

import pickle as pkl

import examples.vumps.matrices as mat
import examples.vumps.vumps as vumps
import examples.vumps.params as params


def runvumps(H, bond_dimension: int, delta_0=0.1,
             out_directory="./vumps_output",
             vumps_params=None,
             heff_params=None,
             env_params=None):
  """
  Performs a vumps simulation of some Hamiltonian H.

  PARAMETERS
  ----------
  H (array, dxdxdxd): The Hamiltonian to be simulated.
  bond_dimension (int): Bond dimension of the MPS.
  delta_0 (float)        : Initial value for the gradient norm. The
                           convergence thresholds of the various solvers at
                           the initial step are proportional to this, via
                           coefficients in the Krylov and solver param dicts.
  out_directory (string) : Output is saved here. The directory is created
                           if it doesn't exist.
  jax_linalg (bool)   : Determines whether Jax or numpy code is used in
                        certain linear algebra calls.
  vumps_params (dict)    : Hyperparameters for the vumps solver. Formed
                           by 'vumps_params'.
  heff_params (dict)     : Hyperparameters for an eigensolve of certain
                           'effective Hamiltonians'. Formed by
                           'krylov_params()'.
  env_params (dict)      : Hyperparameters for a linear solve that finds
                           the effective Hamiltonians. Formed by
                           'solver_params()'.
  """
  out = vumps.vumps(H, bond_dimension, delta_0=delta_0,
                    out_directory=out_directory,
                    vumps_params=vumps_params,
                    heff_params=heff_params,
                    env_params=env_params)
  return out


def vumps_XX(bond_dimension: int, delta_0=0.1,
             out_directory="./vumps", backend="numpy",
             dtype=np.float32,
             vumps_params=None,
             heff_params=None,
             env_params=None):
  """
  Performs a vumps simulation of the XX model,
  H = XX + YY. Parameters are the same as in runvumps.
  """
  H = mat.H_XX(backend=backend, dtype=dtype)
  out = runvumps(H, bond_dimension, delta_0=delta_0,
                 out_directory=out_directory,
                 vumps_params=vumps_params, heff_params=heff_params,
                 env_params=env_params)
  return out


def vumps_ising(J, h, bond_dimension: int, delta_0=0.1,
                out_directory="./vumps", backend="numpy",
                dtype=np.float32,
                vumps_params=None,
                heff_params=None,
                env_params=None):
  """
  Performs a vumps simulation of the XX model,
  H = XX + YY. Parameters are the same as in runvumps.
  """
  H = mat.H_ising(h, J=J, backend=backend, dtype=dtype)
  out = runvumps(H, bond_dimension, delta_0=delta_0,
                 out_directory=out_directory,
                 vumps_params=vumps_params, heff_params=heff_params,
                 env_params=env_params)
  return out


def vumps_from_checkpoint(checkpoint_path, out_directory="./vumps_load",
                          new_vumps_params=None, new_heff_params=None,
                          new_env_params=None):
  """
  Find the ground state of a uniform two-site Hamiltonian
  using Variational Uniform Matrix Product States. This is a gradient
  descent method minimizing the distance between a given MPS and the
  best approximation to the physical ground state at its bond dimension.

  This interface function initializes vumps from checkpointed data.

  PARAMETERS
  ----------
  checkpoint_path (string): Path to the checkpoint .pkl file.
  """
  writer = vumps.make_writer(out_directory)
  with open(checkpoint_path, "rb") as f:
    chk = pkl.load(f)

  H, iter_data, vumps_params, heff_params, env_params, Niter = chk
  if new_vumps_params is not None:
    vumps_params = {**vumps_params, **new_vumps_params}

  if new_heff_params is not None:
    heff_params = {**heff_params, **new_heff_params}

  if new_env_params is not None:
    env_params = {**env_params, **new_env_params}

  out = vumps.vumps_work(H, iter_data, vumps_params, heff_params, env_params,
                         writer, Niter0=Niter)
  return out
