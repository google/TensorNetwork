import copy
import os
import importlib

import tensornetwork as tn

from examples.vumps.writer import Writer
import examples.vumps.params
import examples.vumps.benchmark as benchmark
import examples.vumps.environment as environment
import examples.vumps.contractions as ct
import examples.vumps.vumps_ops as vumps_ops



##########################################################################
# Functions to handle output.
##########################################################################
def ostr(string):
  """
  Truncates to two decimal places.  """
  return '{:1.2e}'.format(string)


def output(writer, Niter, delta, E, dE, norm, timing_data=None):
  """
  Does the actual outputting.
  """
  outstr = "N = " + str(Niter) + "| eps = " + ostr(delta)
  outstr += "| E = " + '{0:1.16f}'.format(E)
  outstr += "| dE = " + ostr(dE)
  # outstr += "| |B2| = " + ostr(B2)
  if timing_data is not None:
    outstr += "| dt= " + ostr(timing_data["Total"])
  writer.write(outstr)

  this_output = [Niter, E, delta, norm]
  writer.data_write(this_output)

  if timing_data is not None:
    writer.timing_write(Niter, timing_data)


def make_writer(outdir=None):
  """
  Initialize the Writer. Creates a directory in the appropriate place, and
  an output file with headers hardcoded here as 'headers'. The Writer,
  defined in writer.py, remembers the directory and will append to this
  file as the simulation proceeds. It can also be used to pickle data,
  notably the final wavefunction.

  PARAMETERS
  ----------
  outdir (string): Path to the directory where output is to be saved. This
                   directory will be created if it does not yet exist.
                   Otherwise any contents with filename collisions during
                   the simulation will be overwritten.

  OUTPUT
  ------
  writer (writer.Writer): The Writer.
  """

  data_headers = ["N", "E", "|B|", "<psi>"]
  if outdir is None:
    return None
  timing_headers = ["N", "Total", "Diagnostics", "Iteration",
                    "Gradient", "HAc", "Hc", "Gauge Match", "Loss",
                    "Environment", "LH", "RH"]
  writer = Writer(outdir, data_headers=data_headers,
                  timing_headers=timing_headers)
  return writer


###############################################################################
# Effective environment.
###############################################################################
def solve_environment(mpslist, delta, fpoints, H, env_params, H_env=None):
  timing = {}
  timing["Environment"] = benchmark.tick()
  if H_env is None:
    H_env = [None, None]

  lh, rh = H_env  # lowercase means 'from previous iteration'

  A_L, _, A_R = mpslist
  rL, lR = fpoints

  timing["LH"] = benchmark.tick()
  LH = environment.solve_for_LH(A_L, H, lR, env_params, delta, oldLH=lh)
  timing["LH"] = benchmark.tock(timing["LH"], dat=LH)

  timing["RH"] = benchmark.tick()
  RH = environment.solve_for_RH(A_R, H, rL, env_params, delta, oldRH=rh)
  timing["RH"] = benchmark.tock(timing["RH"], dat=RH)

  H_env = [LH, RH]
  timing["Environment"] = benchmark.tock(timing["Environment"], dat=RH)
  return (H_env, timing)


###############################################################################
# Gradient.
###############################################################################
def minimum_eigenpair(matvec, mv_args, guess, tol, max_restarts=100,
                      n_krylov=40, reorth=True, n_diag=10, verbose=True):
  eV = guess
  Ax = matvec(eV, *mv_args)
  for _ in range(max_restarts):
    out = tn.linalg.krylov.eigsh_lanczos(matvec, mv_args,
                                         initial_state=eV,
                                         numeig=1,
                                         num_krylov_vecs=n_krylov,
                                         ndiag=n_diag,
                                         reorthogonalize=reorth)
    ev, eV = out
    ev = ev[0]
    eV = eV[0]
    Ax = matvec(eV, *mv_args)
    e_eV = ev * eV
    rho = tn.linalg.linalg.norm(tn.abs(Ax - e_eV))
    err = rho  # / jnp.linalg.norm(e_eV)
    if err <= tol:
      return (ev, eV, err)
  if verbose:
    print("Warning: eigensolve exited early with error=", err)
  return (ev, eV, err)


def minimize_HAc(mpslist, A_C, Hlist, delta, params):
  """
  The dominant (most negative) eigenvector of HAc.
  """
  A_L, _, A_R = mpslist
  tol = params["tol_coef"]*delta
  mv_args = [A_L, A_R, Hlist]
  ev, newA_C, _ = minimum_eigenpair(ct.apply_HAc, mv_args, A_C, tol,
                                    max_restarts=params["max_restarts"],
                                    n_krylov=params["n_krylov"],
                                    reorth=params["reorth"],
                                    n_diag=params["n_diag"])

  return ev, newA_C


def minimize_Hc(mpslist, Hlist, delta, params):
  A_L, C, A_R = mpslist
  tol = params["tol_coef"]*delta
  mv_args = [A_L, A_R, Hlist]
  ev, newC, _ = minimum_eigenpair(ct.apply_Hc, mv_args, C, tol,
                                  max_restarts=params["max_restarts"],
                                  n_krylov=params["n_krylov"],
                                  reorth=params["reorth"],
                                  n_diag=params["n_diag"])
  return ev, newC


def apply_gradient(iter_data, H, heff_krylov_params, gauge_via_svd):
  """
  Apply the MPS gradient.
  """
  timing = {}
  timing["Gradient"] = benchmark.tick()
  mpslist, a_c, _, H_env, delta = iter_data
  LH, RH = H_env
  Hlist = [H, LH, RH]
  timing["HAc"] = benchmark.tick()
  _, A_C = minimize_HAc(mpslist, a_c, Hlist, delta, heff_krylov_params)
  timing["HAc"] = benchmark.tock(timing["HAc"], dat=A_C)

  timing["Hc"] = benchmark.tick()
  _, C = minimize_Hc(mpslist, Hlist, delta, heff_krylov_params)
  timing["Hc"] = benchmark.tock(timing["Hc"], dat=C)

  timing["Gauge Match"] = benchmark.tick()
  A_L, A_R = vumps_ops.gauge_match(A_C, C, svd=gauge_via_svd)
  timing["Gauge Match"] = benchmark.tock(timing["Gauge Match"], dat=A_L)

  timing["Loss"] = benchmark.tick()
  eL = tn.linalg.linalg.norm(A_C - ct.rightmult(A_L, C))
  eR = tn.linalg.linalg.norm(A_C - ct.leftmult(C, A_R))
  delta = max(eL, eR)
  timing["Loss"] = benchmark.tock(timing["Loss"], dat=delta)

  newmpslist = [A_L, C, A_R]
  timing["Gradient"] = benchmark.tock(timing["Gradient"], dat=C)
  return (newmpslist, A_C, delta, timing)


###############################################################################
# Main loop and friends.
###############################################################################
def vumps_approximate_tm_eigs(C):
  """
  Returns the approximate transfer matrix dominant eigenvectors,
  rL ~ C^dag C, and lR ~ C Cdag = rLdag, both trace-normalized.
  """
  rL = C.H @ C
  rL /= tn.trace(rL)
  lR = rL.H
  return (rL, lR)


def vumps_initialization(d: int, chi: int, dtype=None, backend=None):
  """
  Generate a random uMPS in mixed canonical forms, along with the left
  dominant eV L of A_L and right dominant eV R of A_R.

  PARAMETERS
  ----------
  d: Physical dimension.
  chi: Bond dimension.
  dtype: Data dtype of tensors.

  RETURNS
  -------
  mpslist = [A_L, C, A_R]: Arrays. A_L and A_R have shape (d, chi, chi),
                           and are respectively left and right orthogonal.
                           C is the (chi, chi) centre of orthogonality.
  A_C (array, (d, chi, chi)) : A_L @ C. One of the equations vumps minimizes
                               is A_L @ C = C @ A_R = A_C.
  fpoints = [rL, lR] = C^dag @ C and C @ C^dag respectively. Will converge
                       to the left and right environment Hamiltonians.
                       Both are chi x chi.
  """
  A_1 = tn.randn((d, chi, chi), dtype=dtype, backend=backend)
  A_L, _ = tn.linalg.linalg.qr(A_1, non_negative_diagonal=True)
  C, A_R = tn.linalg.linalg.rq(A_L, non_negative_diagonal=True)
  A_C = ct.rightmult(A_L, C)
  L0, R0 = vumps_approximate_tm_eigs(C)
  fpoints = (L0, R0)
  mpslist = [A_L, C, A_R]
  return (mpslist, A_C, fpoints)


def vumps_iteration(iter_data, H, heff_params, env_params, gauge_via_svd):
  """
  One main iteration of VUMPS.
  """
  timing = {}
  timing["Iteration"] = benchmark.tick()
  mpslist, A_C, delta, grad_time = apply_gradient(iter_data, H, heff_params,
                                                  gauge_via_svd)
  timing.update(grad_time)
  fpoints = vumps_approximate_tm_eigs(mpslist[1])
  _, _, _, H_env, _ = iter_data
  H_env, env_time = solve_environment(mpslist, delta, fpoints, H,
                                      env_params, H_env=H_env)
  iter_data = [mpslist, A_C, fpoints, H_env, delta]
  timing.update(env_time)
  timing["Iteration"] = benchmark.tock(timing["Iteration"], dat=H_env[0])
  return (iter_data, timing)


def diagnostics(mpslist, H, oldE):
  """
  Makes a few computations to output during a vumps run.
  """
  t0 = benchmark.tick()
  E = vumps_ops.twositeexpect(mpslist, H)
  dE = abs(E - oldE)
  norm = vumps_ops.mpsnorm(mpslist)
  tf = benchmark.tock(t0, dat=norm)
  return E, dE, norm, tf


def vumps(H, chi: int, delta_0=0.1,
          out_directory="./vumps", backend=None,
          vumps_params=None,
          heff_params=None,
          env_params=None
          ):
  """
  Find the ground state of a uniform two-site Hamiltonian
  using Variational Uniform Matrix Product States. This is a gradient
  descent method minimizing the distance between a given MPS and the
  best approximation to the physical ground state at its bond dimension.

  This interface function initializes vumps from a random initial state.

  PARAMETERS
  ----------
  H (array, (d, d, d, d)): The Hamiltonian whose ground state is to be found.
  chi (int)              : MPS bond dimension.
  delta_0 (float)        : Initial value for the gradient norm. The
                           convergence thresholds of the various solvers at
                           the initial step are proportional to this, via
                           coefficients in the Krylov and solver param dicts.
  out_directory (str)    : Output is saved here.
  backend (str, backend) : The backend.

  The following arguments are bundled together by initialization functions
  in examples.vumps.params.

  vumps_params (dict)    : Hyperparameters for the vumps solver. Formed
                           by 'vumps_params'.
  heff_params (dict)     : Hyperparameters for an eigensolve of certain
                           'effective Hamiltonians'. Formed by
                           'krylov_params()'.
  env_params (dict)      : Hyperparameters for a linear solve that finds
                           the effective Hamiltonians. Formed by
                           'solver_params()'.

  RETURNS
  -------
  """
  if vumps_params is None:
    vumps_params = examples.vumps.params.vumps_params()
  if heff_params is None:
    heff_params = examples.vumps.params.krylov_params()
  if env_params is None:
    env_params = examples.vumps.params.gmres_params()


  writer = make_writer(out_directory)
  d = H.shape[0]
  mpslist, A_C, fpoints = vumps_initialization(d, chi, H.dtype, backend=backend)
  H_env, env_init_time = solve_environment(mpslist, delta_0,
                                           fpoints, H, env_params)
  iter_data = [mpslist, A_C, fpoints, H_env, delta_0]
  writer.write("Initial solve time: " + str(env_init_time["Environment"]))
  out = vumps_work(H, iter_data, vumps_params, heff_params,
                   env_params, writer)
  return out


def vumps_work(H, iter_data, vumps_params, heff_params, env_params, writer,
               Niter0=1):
  """
  Main work loop for vumps. Should be accessed via one of the interface
  functions above.

  PARAMETERS
  ----------
  H

  """
  checkpoint_every = vumps_params["checkpoint_every"]
  max_iter = vumps_params["max_iter"]

  t_total = benchmark.tick()
  # mpslist, A_C, fpoints, H_env, delta
  mpslist, _, _, _, delta = iter_data
  E = vumps_ops.twositeexpect(mpslist, H)
  writer.write("vuMPS, a love story.")
  writer.write("Initial energy: " + str(E))
  writer.write("And so it begins...")
  for Niter in range(Niter0, vumps_params["max_iter"]+Niter0):
    dT = benchmark.tick()
    timing = {}
    oldE = E
    iter_data, iter_time = vumps_iteration(iter_data, H, heff_params,
                                           env_params,
                                           vumps_params["gauge_via_svd"])
    mpslist, _, _, _, _ = iter_data
    timing.update(iter_time)

    E, dE, norm, tD = diagnostics(mpslist, H, oldE)
    timing["Diagnostics"] = tD
    timing["Total"] = benchmark.tock(dT, dat=iter_data[1])
    output(writer, Niter, delta, E, dE, norm, timing)

    if delta <= vumps_params["gradient_tol"]:
      writer.write("Convergence achieved at iteration " + str(Niter))
      break

    if checkpoint_every is not None and (Niter+1) % checkpoint_every == 0:
      writer.write("Checkpointing...")
      to_pickle = [H, iter_data, vumps_params, heff_params, env_params]
      to_pickle.append(Niter)
      writer.pickle(to_pickle, Niter)

  if Niter == max_iter - 1:
    writer.write("Maximum iteration " + str(max_iter) + " reached.")
  t_total = benchmark.tock(t_total, dat=mpslist[0])
  writer.write("The main loops took " + str(t_total) + " seconds.")
  writer.write("Simulation finished. Pickling results.")
  to_pickle = [H, iter_data, vumps_params, heff_params, env_params, Niter]
  writer.pickle(to_pickle, Niter)
  return (iter_data, timing)
