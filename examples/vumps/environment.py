import examples.vumps.contractions as ct
import examples.vumps.vumps_ops as vumps_ops
import tensornetwork as tn


def call_solver(matvec, matvec_args, hI, params, x0, tol):
  """
  Code used by both solve_for_RH and solve_for_LH to call the
  sparse solver. Solves matvec(*matvec_args, x) = hI for x.

  PARAMETERS
  ----------
  matvec (jax.tree_util.Partial): A function implementing the linear
                                  transform.
  matvec_args (list)            : List of constant arguments to matvec.
  hI (array)                    : The right hand side of the equation.
  params (dict)                 : Parameters for the solver.
  x0 (array)                    : Initial guess vector.
  tol (float)                   : Convergence threshold.

  RETURNS
  ------
  """
  if x0 is None:
    x0 = tn.randn(hI.shape, dtype=hI.dtype, backend=hI.backend)

  solution, _ = tn.linalg.krylov.gmres(matvec,
                                       hI,
                                       A_args=matvec_args,
                                       x0=x0,
                                       tol=tol,
                                       num_krylov_vectors=params["n_krylov"],
                                       maxiter=params["max_restarts"])
  return solution


###############################################################################
# LH
###############################################################################

#TODO: jit
def LH_matvec(lR, A_L, v):
  chi = A_L.shape[2]
  v = v.reshape((chi, chi))
  Th_v = ct.XopL(A_L, v)
  vR = ct.proj(v, lR)*tn.eye(chi, dtype=A_L.dtype, backend=A_L.backend)
  v = v - Th_v + vR
  v = v.flatten()
  return v


#TODO: jit
def prepare_for_LH_solve(A_L, H, lR):
  """
  Computes A and b in the A x = B to be solved for the left environment
  Hamiltonian. Separates things that can be Jitted from the rest of
  solve_for_LH.
  """
  hL_bare = ct.compute_hL(A_L, H)
  hL_div = ct.proj(hL_bare, lR)*tn.eye(hL_bare.shape[0], dtype=A_L.dtype,
                                       backend=A_L.backend)
  hL = hL_bare - hL_div
  return hL


def solve_for_LH(A_L, H, lR, params, delta, oldLH=None):
  """
  Find the renormalized left environment Hamiltonian using a sparse
  solver.
  """
  tol = params["tol_coef"]*delta
  hL = prepare_for_LH_solve(A_L, H, lR)
  matvec_args = [lR, A_L]
  LH, _ = tn.linalg.krylov.gmres(LH_matvec,
                                 hL,
                                 A_args=matvec_args,
                                 x0=oldLH,
                                 tol=tol,
                                 num_krylov_vectors=params["n_krylov"],
                                 maxiter=params["max_restarts"])
  LH = call_solver(LH_matvec, matvec_args, hL, params, oldLH, tol)
  return LH


###############################################################################
# RH
###############################################################################
#@jax.tree_util.Partial
#@jax.jit
def RH_matvec(rL, A_R, v):
  chi = A_R.shape[2]
  v = v.reshape((chi, chi))
  Th_v = ct.XopR(A_R, v)
  Lv = ct.proj(rL, v)*tn.eye(chi, dtype=A_R.dtype, backend=A_R.backend)
  v = v - Th_v + Lv
  v = v.flatten()
  return v


def solve_for_RH(A_R, H, rL, params, delta, oldRH=None):
  """
  Find the renormalized right environment Hamiltonian using a sparse
  solver.
  """
  tol = params["tol_coef"]*delta
  hR_bare = ct.compute_hR(A_R, H)
  hR_div = ct.proj(rL, hR_bare)*tn.eye(hR_bare.shape[0], dtype=A_R.dtype,
                                       backend=A_R.backend)
  hR = hR_bare - hR_div
  matvec_args = [rL, A_R]
  RH, _ = tn.linalg.krylov.gmres(RH_matvec,
                                 hR,
                                 A_args=matvec_args,
                                 x0=oldRH,
                                 tol=tol,
                                 num_krylov_vectors=params["n_krylov"],
                                 maxiter=params["max_restarts"])
  RH = call_solver(RH_matvec, matvec_args, hR, params, oldRH, tol)
  return RH
