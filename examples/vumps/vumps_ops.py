"""
Functions that operate upon MPS tensors as if they were matrices.
These functions are not necessarily backend-agnostic.
"""
import time
from functools import partial
import examples.vumps.contractions as ct
import tensornetwork as tn


def frobnorm(A, B=None):
  """
  The Frobenius norm of the difference between A and B, divided by the
  number of entries in A.
  """
  if B is None:
    B = tn.zeros(A.shape, backend=A.backend)
  ans = (1./A.size)*tn.linalg.linalg.norm(tn.abs(A.ravel()-B.ravel()))
  return ans


#TODO:jit
def twositeexpect(mpslist, H):
  """
  The expectation value of the operator H in the state represented
  by A_L, C, A_R in mpslist.

  RETURNS
  -------
  out: The expectation value.
  """
  A_L, C, A_R = mpslist
  A_CR = ct.leftmult(C, A_R)
  expect = ct.twositeexpect(A_L, A_CR, H)
  return expect


#TODO: jit
def mpsnorm(mpslist):
  A_L, C, A_R = mpslist
  A_CR = ct.leftmult(C, A_R)
  rho = ct.rholoc(A_L, A_CR)
  the_norm = tn.abs(tn.trace(rho))
  return the_norm


#TODO:jit
def gauge_match(A_C, C, svd=True):
  """
  Return approximately gauge-matched A_L and A_R from A_C and C
  using a polar decomposition.

  A_L and A_R are chosen to minimize ||A_C - A_L C|| and ||A_C - C A_R||.
  The respective solutions are the isometric factors in the
  polar decompositions of A_C C\dag and C\dag A_C.

  PARAMETERS
  ----------
  A_C (d, chi, chi)
  C (chi, chi)     : MPS tensors.
  svd (bool)      :  Toggles whether the SVD or QDWH method is used for the 
                     polar decomposition. In general, this should be set
                     False on the GPU and True otherwise.

  RETURNS
  -------
  A_L, A_R (d, chi, chi): Such that A_L C A_R minimizes ||A_C - A_L C||
                          and ||A_C - C A_R||, with A_L and A_R
                          left (right) isometric.
  """
  UC = polarU_SVD(C)
  UAc_l = polarU_SVD(A_C, pivot_axis=2)
  A_L = UAc_l @ UC.H
  UAc_r = polarU_SVD(A_C, pivot_axis=1)
  A_R = UC.H @  UAc_r
  return (A_L, A_R)

#############################################################################
# Polar decomposition
#############################################################################
#TODO: jit
def polarU_SVD(A, pivot_axis=1):
  """
  Compute the unitary part of the polar decomposition explitly as
  U = u @ vH where A = u @ S @ vh is the SVD of A. This is about twice
  as fast as polarU_QDWH on the
  CPU or TPU but around an order of magnitude slower on the GPU.
  """
  w, _, vh, _ = tn.linalg.linalg.svd(A, pivot_axis=pivot_axis)
  u = w @ vh
  return u
