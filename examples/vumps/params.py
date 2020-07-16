"""
Dictionaries bundling together parameters for vumps runs.
"""


def krylov_params(n_krylov=40, n_diag=100, tol_coef=0.01, max_restarts=30,
                  reorth=True):
    """
    Bundles parameters for the Lanczos eigensolver. These control
    the expense of finding the left and right environment tensors, and of
    minimizing the effective Hamiltonians.

    PARAMETERS
    ----------
    n_krylov (int, 40): Size of the Krylov subspace.
    n_diag (int, 100) : The solver checks convergence at this periodicity.
    tol_coef (float, 0.01): This number times the MPS gradient will be the
                            convergence threshold of the eigensolve.
    max_restarts (int, 30): The solver exits here even if not yet converged.
    reorth (bool, True): If True the solver reorthogonalizes the Lanczos
                         vectors at each iteration. This is more expensive,
                         especially for large n_krylov and low chi,
                         but may be necessary for vumps to converge.
    """
    return {"n_krylov": n_krylov, "n_diag": n_diag, "reorth": reorth,
            "tol_coef": tol_coef, "max_restarts": max_restarts}


def gmres_params(n_krylov=40, max_restarts=20, tol_coef=0.01):
    """
    Bundles parameters for the GMRES linear solver. These control the
    expense of finding the left and right environment Hamiltonians.

    PARAMETERS
    ----------
    n_krylov (int): Size of the Krylov subspace.
    max_restarts (int): Maximum number of times to iterate the Krylov
                        space construction.
    tol_coef (float): This number times the MPS gradient will set the
                      convergence threshold of the linear solve.


    RETURNS
    -------
    A dictionary storing each of these parameters.
    """
    return {"solver": "gmres", "n_krylov": n_krylov,
            "max_restarts": max_restarts, "tol_coef": tol_coef}


def lgmres_params(inner_m=30, outer_k=3, maxiter=100, tol_coef=0.01):
    """
    Bundles parameters for the LGMRES linear solver. These control the
    expense of finding the left and right environment Hamiltonians.

    PARAMETERS
    ----------
    inner_m (int, 30): Number of gmres iterations per outer k loop.
    outer_k (int, 3) : Number of vectors to carry between inner iterations.
    maxiter (int)    : lgmres terminates after this many iterations.
    tol_coef (float): This number times the MPS gradient will set the
                      convergence threshold of the linear solve.

    RETURNS
    -------
    A dictionary storing each of these parameters.
    """
    return {"solver": "lgmres", "inner_m": inner_m, "maxiter": maxiter,
            "outer_k": outer_k, "tol_coef": tol_coef}


def vumps_params(checkpoint_every=500, gauge_via_svd=True, gradient_tol=1E-3,
                 max_iter=200):
    """
    Bundles parameters for the vumps solver itself.

    PARAMETERS
    ----------
    gradient_tol (float)   : Convergence is declared once the gradient norm is
                             at least this small.
    max_iter (int)         : VUMPS ends after this many iterations even if
                             unconverged.
    checkpoint_every (int) : Simulation data is pickled at this periodicity.
    out_directory (string) : Output is saved here. The directory is created
                             if it doesn't exist.
    gauge_via_svd (bool, True): With the Jax backend, toggles whether the gauge
                                match at the
                                end of each iteration is computed using
                                an SVD or the QDWH-based polar decomposition.
                                The former is typically faster on the CPU
                                or TPU, but the latter is much faster on the
                                GPU. With the NumPy backend, this
                                parameter has no effect and the SVD is always
                                used.

    RETURNS
    -------
    A dictionary storing each of these parameters.
    """

    return {"checkpoint_every": checkpoint_every,
            "gauge_via_svd": gauge_via_svd,
            "gradient_tol": gradient_tol, "max_iter": max_iter}
