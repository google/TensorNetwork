import jax
import jax.numpy as jnp

import tensornetwork as tn

import jax_vumps.jax_backend.contractions as ct
# import jax_vumps.jax_backend.lanczos as lz

TN_JAX_BACKEND = tn.backends.backend_factory.get_backend('jax')


def minimum_eigenpair(matvec, mv_args, guess, tol, max_restarts=100,
                      n_krylov=40, reorth=True, n_diag=10, verbose=True,
                      A_norm=None):
    eV = guess
    Ax = matvec(eV, *mv_args)
    tol = max(tol, jnp.finfo(guess.dtype).eps)
    tolarr = jnp.array(tol)
    for j in range(max_restarts):
        out = TN_JAX_BACKEND.eigsh_lanczos(matvec, mv_args,
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
        rho = jnp.linalg.norm(jnp.abs(Ax - e_eV))
        err = rho  # / jnp.linalg.norm(e_eV)
        if err <= tol:
            return (ev, eV, err)
    if verbose:
        print("Warning: eigensolve exited early with error=", err)
    return (ev, eV, err)

###############################################################################
# Effective Hamiltonians for A_C.
###############################################################################

#  @jax.tree_util.Partial
#  @jax.jit
#  def apply_HAc_for_solver(A_L, A_R, Hlist, v):
#      A_C = v.reshape(A_L.shape)
#      newA_C = ct.apply_HAc(A_C, A_L, A_R, Hlist)
#      newv = newA_C.flatten()
#      return newv

#  def minimize_HAc(mpslist, A_C, Hlist, delta, params):
#      """
#      The dominant (most negative) eigenvector of HAc.
#      """
#      A_L, C, A_R = mpslist
#      tol = params["tol_coef"]*delta
#      lzout = lz.minimum_eigenpair(apply_HAc_for_solver, [A_L, A_R, Hlist],
#                                    params["n_krylov"],
#                                    maxiter=params["max_restarts"], tol=tol,
#                                    v0=A_C.flatten())
#      ev, newA_C, err = lzout
#      newA_C = newA_C.reshape((A_C.shape))
#      return ev, newA_C


#  @jax.tree_util.Partial
#  @jax.jit
#  def apply_HAc_for_solver(A_L, A_R, Hlist, A_C):
#      newC = ct.apply_Hc(A_C, A_L, A_R, Hlist)
#      return newC


def minimize_HAc(mpslist, A_C, Hlist, delta, params):
    """
    The dominant (most negative) eigenvector of HAc.
    """
    A_L, C, A_R = mpslist
    tol = params["tol_coef"]*delta
    mv_args = [A_L, A_R, Hlist]
    ev, newA_C, err = minimum_eigenpair(ct.apply_HAc, mv_args, A_C, tol,
                                        max_restarts=params["max_restarts"],
                                        n_krylov=params["n_krylov"],
                                        reorth=params["reorth"],
                                        n_diag=params["n_diag"])

    # newA_C = newA_C.reshape((A_C.shape))
    return ev, newA_C


###############################################################################
# Effective Hamiltonians for C.
###############################################################################

#  @jax.tree_util.Partial
#  @jax.jit
#  def apply_Hc_for_solver(A_L, A_R, Hlist, v):
#      chi = A_L.shape[2]
#      C = v.reshape((chi, chi))
#      newC = ct.apply_Hc(C, A_L, A_R, Hlist)
#      newv = newC.flatten()
#      return newv


#  def minimize_Hc(mpslist, Hlist, delta, params):
#      """
#      The dominant (most negative) eigenvector of Hc.
#      """
#      A_L, C, A_R = mpslist
#      tol = params["tol_coef"]*delta
#      lzout = lz.minimum_eigenpair(apply_Hc_for_solver, [A_L, A_R, Hlist],
#                                    params["n_krylov"],
#                                    maxiter=params["max_restarts"], tol=tol,
#                                    v0=C.flatten())
#      ev, newC, err = lzout
#      newC = newC.reshape((C.shape))
#      return ev, newC


#  @jax.tree_util.Partial
#  @jax.jit
#  def apply_Hc_for_solver(A_L, A_R, Hlist, C):
#      newC = ct.apply_Hc(C, A_L, A_R, Hlist)
#      return newC


def minimize_Hc(mpslist, Hlist, delta, params):
    A_L, C, A_R = mpslist
    tol = params["tol_coef"]*delta
    mv_args = [A_L, A_R, Hlist]
    ev, newC, err = minimum_eigenpair(ct.apply_Hc, mv_args, C, tol,
                                      max_restarts=params["max_restarts"],
                                      n_krylov=params["n_krylov"],
                                      reorth=params["reorth"],
                                      n_diag=params["n_diag"],
                                      A_norm=None)
    return ev, newC
