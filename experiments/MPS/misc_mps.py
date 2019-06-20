# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""miscellaneous functions needed for Matrix Product States simulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensornetwork import ncon as ncon_tn
import copy
import functools as fct
import tensornetwork as tn
from scipy.sparse.linalg import LinearOperator, lgmres, eigs
ncon_defuned = tf.contrib.eager.defun(ncon_tn)


def transfer_op(As, Bs, direction, x):
  """
    (mixed) transfer operator for a list of mps tensors

    Args:
        As,Bs (list of tf.Tensor):     the mps tensors (Bs are on the conjugated side)
        direction (int or str):        can be (1,'l','left') or (-1,'r','right) for left or right 
                                       transfer operation
        x (tf.Tensor):                 input matrix
    Returns:
        tf.Tensor:  the evolved matrix
    """

  if direction in ('l', 'left', 1):
    for n in range(len(As)):
      x = ncon([x, As[n], tf.conj(Bs[n])], [(0, 1), (0, 2, -1), (1, 2, -2)])
  elif direction in ('r', 'right', -1):
    for n in reversed(range(len(As))):
      x = ncon([x, As[n], tf.conj(Bs[n])], [(0, 1), (-1, 2, 0), (-2, 2, 1)])
  else:
    raise ValueError("Invalid direction: {}".format(direction))

  return x


transfer_op_defuned = tf.contrib.eager.defun(transfer_op)


def add_layer(B, mps_tensor, mpo, conj_mps_tensor, direction):
  """
    adds an mps-mpo-mps layer to a left or right block "E"; used in dmrg to calculate the left and right
    environments
    Args
        B (tf.Tensor):               a tensor of shape (D1,D1',M1) (for direction>0) or (D2,D2',M2) (for direction>0)
        mps_tensor (tf.Tensor):      tensor of shape =(Dl,Dr,d)
        mpo_tensor (tf.Tensor):      tensor of shape = (Ml,Mr,d,d')
        conj_mps_tensor (tf.Tensor): tensor of shape =(Dl',Dr',d')
                                     the mps tensor on the conjugated side
                                     this tensor is complex-conjugated inside the routine; usually, the user will like to pass 
                                     the unconjugated tensor
        direction (int or str):      direction in (1,'l','left'): add a layer to the right of `B`
                                     direction in (-1,'r','right'): add a layer to the left of `B`
    Returns:
        tf.Tensor of shape (Dr,Dr',Mr) for direction in (1,'l','left')
        tf.Tensor of shape (Dl,Dl',Ml) for direction in (-1,'r','right')
    """
  if direction in ('l', 'left', 1):
    return ncon(
        [B, mps_tensor, mpo, tf.conj(conj_mps_tensor)],
        [[1, 4, 3], [1, 2, -1], [3, -3, 5, 2], [4, 5, -2]])

  if direction in ('r', 'right', -1):
    return ncon(
        [B, mps_tensor, mpo, tf.conj(conj_mps_tensor)],
        [[1, 4, 3], [-1, 2, 1], [-3, 3, 5, 2], [-2, 5, 4]])


add_layer_defuned = tf.contrib.eager.defun(add_layer)


def one_minus_pseudo_unitcell_transfer_op(direction, mps, left_dominant,
                                          right_dominant, vector):
  """
    calculates action of 11-Transfer-Operator +|r)(l|
    Args:
        direction (int or str):          if (1,'l','left'): do left multiplication
                                         if (-1,'r','right'): do right multiplication
        mps (InfiniteMPSCentralGauge):   an infinite mps
        left_dominant (tf.Tensor):       tensor of shape (mps.D[0],mps.D[0])
                                         left dominant eigenvector of the unit-cell transfer operator of mps
        right_dominant (tf.Tensor):      tensor of shape (mps.D[-1],mps.D[-1])
                                         right dominant eigenvector of the unit-cell transfer operator of mps
        vector (tf.Tensor):              tensor of shape (mps.D[0]*mps.D[0]) or (mps.D[-1]*mps.D[-1])
                                         the input vector
    Returns:
        tf.Tensor of shape (mps.D[0]*mps.D[0]) or (mps.D[-1]*mps.D[-1])
    """

  if direction in (1, 'l', 'left'):
    x = tf.reshape(tf.convert_to_tensor(vector), (mps.D[0], mps.D[0]))
    temp = x - mps.unitcell_transfer_op(
        'left', x) + ncon([x, right_dominant], [[1, 2], [1, 2]]) * left_dominant
    return tf.reshape(temp, [mps.D[-1] * mps.D[-1]]).numpy()

  if direction in (-1, 'r', 'right'):
    x = tf.reshape(tf.convert_to_tensor(vector), [mps.D[-1], mps.D[-1]])
    temp = x - mps.unitcell_transfer_op(
        'right',
        x) + ncon([left_dominant, x], [[1, 2], [1, 2]]) * right_dominant
    return tf.reshape(temp, [mps.D[0] * mps.D[0]]).numpy()


def LGMRES_solver(mps,
                  direction,
                  left_dominant,
                  right_dominant,
                  inhom,
                  x0,
                  precision=1e-10,
                  nmax=2000,
                  **kwargs):
  """
    see Appendix of arXiv:1801.02219 for details of this
    This routine uses scipy's sparse.lgmres module. tf.Tensors are mapped to numpy 
    and back to tf.Tensor for each application of the sparse matrix vector product.
    This is not optimal and will be improved in a future version
    Args:
        mps (InfiniteMPSCentralGauge):   an infinite mps
        direction (int or str):          if (1,'l','left'): do left multiplication
                                         if (-1,'r','right'): do right multiplication
        left_dominant (tf.Tensor):       tensor of shape (mps.D[0],mps.D[0])
                                         left dominant eigenvector of the unit-cell transfer operator of mps
        right_dominant (tf.Tensor):      tensor of shape (mps.D[-1],mps.D[-1])
                                         right dominant eigenvector of the unit-cell transfer operator of mps
        inhom (tf.Tensor):               vector of shape (mps.D[0]*mps.D[0]) or (mps.D[-1]*mps.D[-1])
    Returns:
        tf.Tensor
    """
  #mps.D[0] has to be mps.D[-1], so no distincion between direction='l' or direction='r' has to be made here
  if not tf.equal(mps.D[0], mps.D[-1]):
    raise ValueError(
        'in LGMRES_solver: mps.D[0]!=mps.D[-1], can only handle intinite MPS!')
  inhom_numpy = tf.reshape(inhom, [mps.D[0] * mps.D[0]]).numpy()
  x0_numpy = tf.reshape(x0, [mps.D[0] * mps.D[0]]).numpy()
  mv = fct.partial(one_minus_pseudo_unitcell_transfer_op,
                   *[direction, mps, left_dominant, right_dominant])

  LOP = LinearOperator((int(mps.D[0])**2, int(mps.D[-1])**2),
                       matvec=mv,
                       dtype=mps.dtype.as_numpy_dtype)
  out, info = lgmres(
      A=LOP, b=inhom_numpy, x0=x0_numpy, tol=precision, maxiter=nmax, **kwargs)

  return tf.reshape(tf.convert_to_tensor(out), [mps.D[0], mps.D[0]]), info


def compute_steady_state_Hamiltonian_GMRES(direction,
                                           mps,
                                           mpo,
                                           left_dominant,
                                           right_dominant,
                                           precision=1E-10,
                                           nmax=1000):
  """
    calculates the left or right Hamiltonain environment of an infinite MPS-MPO-MPS network
    This routine uses scipy's sparse.lgmres module. tf.Tensors are mapped to numpy 
    and back to tf.Tensor for each application of the sparse matrix vector product.
    This is not optimal and will be improved in a future version

    Args:
        direction (int or str):        if (1,'l','left'): obtain left environment
                                       if (-1,'r','right'): obtain right environment
        mps (InfiniteMPSCentralGauge): an infinite mps
        mpo (InfiniteMPO):             the mpo
        left_dominant (tf.tensor):     tensor of shape (mps.D[0],mps.D[0])
                                       left dominant eigenvvector of the unit-cell transfer operator of mps
        right_dominant (tf.Tensor):    tensor of shape (mps.D[-1],mps.D[-1])
                                       right dominant eigenvvector of the unit-cell transfer operator of mps
        precision (float):             desired precision of the environments
        nmax (int):                    maximum iteration numner
    Returns
        H (tf.Tensor):   tensor of shape (mps.D[0],mps.D[0],mpo.D[0])
                         Hamiltonian environment
        h (tf.Tensor):   tensor of shape (1)
                         average energy per unitcell 
    """

  dummy1 = mpo.get_boundary_vector('l')
  dummy2 = mpo.get_boundary_vector('r')

  if direction in (1, 'l', 'left'):
    L = ncon([
        mps.get_tensor(-1),
        mpo.get_boundary_mpo('left'),
        tf.conj(mps.get_tensor(-1))
    ], [[1, 2, -1], [-3, 3, 2], [1, 3, -2]])
    for n in range(len(mps)):
      L = add_layer(
          L,
          mps.get_tensor(n),
          mpo.get_tensor(n),
          mps.get_tensor(n),
          direction='l')

    h = ncon([L, dummy2, right_dominant], [[1, 2, 3], [3], [1, 2]])
    inhom = ncon([L, dummy2], [[-1, -2, 1], [1]]) - h * tf.diag(
        tf.ones([mps.D[-1]], dtype=mps.dtype))
    [out, info] = LGMRES_solver(
        mps=mps,
        direction=direction,
        left_dominant=left_dominant,
        right_dominant=right_dominant,
        inhom=inhom,
        x0=tf.zeros([mps.D[0], mps.D[0]], dtype=mps.dtype),
        precision=precision,
        nmax=nmax)
    temp = L.numpy()
    temp[:, :, 0] = out.numpy()
    return tf.convert_to_tensor(temp), h

  if direction in (-1, 'r', 'right'):
    R = ncon([
        mps.get_tensor(0),
        mpo.get_boundary_mpo('right'),
        tf.conj(mps.get_tensor(0))
    ], [[-1, 2, 1], [-3, 3, 2], [-2, 3, 1]])
    for n in reversed(range(len(mps))):
      R = add_layer(
          R,
          mps.get_tensor(n),
          mpo.get_tensor(n),
          mps.get_tensor(n),
          direction='r')
    h = ncon([dummy1, left_dominant, R], [[3], [1, 2], [1, 2, 3]])
    inhom = ncon([dummy1, R], [[1], [-1, -2, 1]]) - h * tf.diag(
        tf.ones([mps.D[0]], dtype=mps.dtype))
    [out, info] = LGMRES_solver(
        mps=mps,
        direction=direction,
        left_dominant=left_dominant,
        right_dominant=right_dominant,
        inhom=inhom,
        x0=tf.zeros([mps.D[0], mps.D[0]], dtype=mps.dtype),
        precision=precision,
        nmax=nmax)

    temp = R.numpy()
    temp[:, :, -1] = out.numpy()
    return tf.convert_to_tensor(temp), h


def compute_Hamiltonian_environments(mps,
                                     mpo,
                                     precision=1E-10,
                                     precision_canonize=1E-10,
                                     nmax=1000,
                                     nmax_canonize=10000,
                                     ncv=40,
                                     numeig=6,
                                     pinv=1E-30):
  """
    calculates the Hamiltonian environments of an infinite MPS-MPO-MPS network
    This routine uses scipy's sparse.lgmres module. tf.Tensors are mapped to numpy 
    and back to tf.Tensor for each application of the sparse matrix vector product.
    This is not optimal and will be improved in a future version

    Args:
        mps (InfiniteMPSCentralGauge):    an infinite mps
        mpo (InfiniteMPO):                an infinite mpo
        precision (float):                desired precision of the environments
        precision_canonize (float):       desired precision for mps canonization
        nmax (int):                       maximum iteration numner
        nmax_canonize (int):              maximum iteration number in TMeigs during canonization
        ncv (int):                        number of krylov vectors in TMeigs during canonization
        numeig (int):                     number of eigenvectors targeted by sparse soler in TMeigs during canonization
        pinv (float):                     pseudo inverse threshold during canonization

    Returns:
        lb (tf.Tensor):  tensor of shape (mps.D[0],mps.D[0],mpo.D[0])
                         left Hamiltonian environment, including coupling of unit-cell to the left environment
        rb (tf.Tensor):  tensor of shape (mps.D[-1],mps.D[-1],mpo.D[-1])
                         right Hamiltonian environment, including coupling of unit-cell to the right environment
        hl (tf.Tensor):  tensor of shape(1)
                         average energy per left unitcell 
        hr (tf.Tensor):  tensor of shape(1)
                         average energy per right unitcell 
    NOTE:  hl and hr do not have to be identical
    """

  mps.restore_form(
      precision=precision_canonize,
      ncv=ncv,
      nmax=nmax_canonize,
      numeig=numeig,
      pinv=pinv)
  mps.position(len(mps))
  lb, hl = compute_steady_state_Hamiltonian_GMRES(
      'l',
      mps,
      mpo,
      left_dominant=tf.diag(tf.ones(mps.D[-1], dtype=mps.dtype)),
      right_dominant=ncon([mps.mat, tf.conj(mps.mat)], [[-1, 1], [-2, 1]]),
      precision=precision,
      nmax=nmax)
  rmps = mps.get_right_orthogonal_imps(
      precision=precision_canonize,
      ncv=ncv,
      nmax=nmax_canonize,
      numeig=numeig,
      pinv=pinv,
      restore_form=False)
  rb, hr = compute_steady_state_Hamiltonian_GMRES(
      'r',
      rmps,
      mpo,
      right_dominant=tf.diag(tf.ones(mps.D[0], dtype=mps.dtype)),
      left_dominant=ncon([mps.mat, tf.conj(mps.mat)], [[1, -1], [1, -2]]),
      precision=precision,
      nmax=nmax)
  return lb, rb, hl, hr


def HA_product(L, mpo, R, mps):
  """
    the local matrix vector product of the DMRG optimization
    Args:
        L (tf.Tensor):    left environment of the local sites
        mpo (tf.Tensor):  local mpo tensor
        R (tf.Tensor):    right environment of the local sites
        mps (tf.Tensor):  local mps tensor
    Returns:
        tf.Tensor:   result of the local contraction
    """
  return ncon([L, mps, mpo, R],
              [[1, -1, 2], [1, 3, 4], [2, 5, -2, 3], [4, -3, 5]])


def prepare_tensor_QR(tensor, direction):
  """
    prepares an mps tensor using svd decomposition 
    Args
        tensor (tf.Tensors): tensor of shape(D1,D2,d)
                             an mps tensor
    direction (int):         if `int` > 0 returns left orthogonal decomposition, 
                             if `int` < 0 returns right orthogonal decomposition
    Returns:
        if direction>0:     
        (out,s,v)
         out (tf.Tensor): a left isometric tf.Tensor of dimension (D1,D,d)
         s (tf.Tensor):   the singular values of length D
         v (tf.Tensor):   a right isometric tf.Tensor of dimension (D,D2)
        if direction<0:     
        (u,s,out)
        u (tf.Tensor):    a left isometric tf.Tensor of dimension (D1,D)
        s (tf.Tensor):    the singular values of length D
        out (tf.Tensor):  a right isometric tf.Tensor of dimension (D,D2,d)
    """
  l1, d, l2 = tf.unstack(tf.shape(tensor))
  if direction in ('l', 'left', 1):
    temp = tf.reshape(tensor, [d * l1, l2])
    q, r = tf.linalg.qr(temp)
    Z = tf.linalg.norm(r)
    r /= Z
    size1, size2 = tf.unstack(tf.shape(q))
    out = tf.reshape(q, [l1, d, size2])
    return out, r, Z

  if direction in ('r', 'right', -1):
    temp = tf.reshape(tensor, [l1, d * l2])
    q, r = tf.linalg.qr(tf.transpose(tf.conj(temp)))
    Z = tf.linalg.norm(r)
    r /= Z
    size1, size2 = tf.unstack(tf.shape(q))
    out = tf.reshape(tf.transpose(tf.conj(q)), [size2, d, l2])

    return tf.transpose(tf.conj(r)), out, Z


prepare_tensor_QR_defuned = tf.contrib.eager.defun(prepare_tensor_QR)


def prepare_tensor_SVD(tensor, direction):
  """
    prepares an mps tensor using svd decomposition 
    Args:
        tensor (tf.Tensors):  tensor of shape(D1,D2,d)
                              an mps tensor
        direction (int):      if `int` > 0: returns left orthogonal decomposition, 
                              if `int` < 0: returns right orthogonal decomposition

    Returns:
        if direction>0: (out,s,v) with
        out (tf.Tensor): a left isometric tf.Tensor of dimension (D1,D,d)
        s (tf.Tensor):   the singular values of length D
        v (tf.Tensor):   a right isometric tf.Tensor of dimension (D,D2)

        if direction<0: (u,s,out) with
        u (tf.Tensor):   a left isometric tf.Tensor of dimension (D1,D)
        s (tf.Tensor):   the singular values of length D
        out (tf.Tensor): a right isometric tf.Tensor of dimension (D,D2,d)
    """
  l1, d, l2 = tf.unstack(tf.shape(tensor))

  if direction in ('l', 'left', 1):
    temp = tf.reshape(tensor, [d * l1, l2])
    s, u, v = tf.linalg.svd(temp, full_matrices=False)
    Z = tf.linalg.norm(s)
    #s/=Z
    size1, size2 = tf.unstack(tf.shape(u))
    out = tf.reshape(u, [l1, d, size2])
    return out, s, tf.transpose(tf.conj(v)), Z

  if direction in ('r', 'right', -1):
    temp = tf.reshape(tensor, [l1, d * l2])
    s, u, v = tf.linalg.svd(temp, full_matrices=False)
    Z = tf.linalg.norm(s)
    #s/=Z
    size1, size2 = tf.unstack(tf.shape(v))
    out = tf.reshape(tf.transpose(tf.conj(v)), [size2, d, l2])
    return u, s, out, Z


prepare_tensor_SVD_defuned = tf.contrib.eager.defun(prepare_tensor_SVD)


def apply_2site_schmidt_canonical(op,
                                  L0,
                                  G1,
                                  L1,
                                  G2,
                                  L2,
                                  max_bond_dim=None,
                                  auto_trunc_max_err=0.0):
  """
  Applies a two-site local operator to an MPS.
  Takes Lambda and Gamma tensors (Schmidt canonical form)
  and returns new ones, as well as the new norm of the state.
  """
  if tf.executing_eagerly():
    # FIXME: Not ideal, but these ops are very costly at compile time
    op_shp = tf.shape(op)
    tf.assert_equal(
        tf.shape(G1)[1],
        op_shp[2],
        message="Operator dimensions do not match MPS physical dimensions.")
    tf.assert_equal(
        tf.shape(G2)[1],
        op_shp[3],
        message="Operator dimensions do not match MPS physical dimensions.")

  # TODO(ash): Can we assume these are diagonal?
  L0_i = tf.matrix_inverse(L0)
  L2_i = tf.matrix_inverse(L2)

  net = tn.TensorNetwork()
  nL0_i = net.add_node(L0_i, axis_names=["L", "R"])
  nL0 = net.add_node(L0, axis_names=["L", "R"])
  nG1 = net.add_node(G1, axis_names=["L", "p", "R"])
  nL1 = net.add_node(L1, axis_names=["L", "R"])
  nG2 = net.add_node(G2, axis_names=["L", "p", "R"])
  nL2 = net.add_node(L2, axis_names=["L", "R"])
  nL2_i = net.add_node(L2_i, axis_names=["L", "R"])
  nop = net.add_node(op, axis_names=["p_out_1", "p_out_2", "p_in_1", "p_in_2"])

  b0 = net.connect(nL0_i["R"], nL0["L"])
  b1 = net.connect(nL0["R"], nG1["L"])
  b2 = net.connect(nG1["R"], nL1["L"])
  b3 = net.connect(nL1["R"], nG2["L"])
  b4 = net.connect(nG2["R"], nL2["L"])
  b5 = net.connect(nL2["R"], nL2_i["L"])

  net.connect(nG1["p"], nop["p_in_1"])
  net.connect(nG2["p"], nop["p_in_2"])

  output_order = [nL0["L"], nop["p_out_1"], nop["p_out_2"], nL2["R"]]
  net.contract(b1)
  net.contract(b2)
  net.contract(b3)
  n_mps = net.contract(b4)
  n_block = net.contract_between(nop, n_mps)

  nu, ns, nvh, s_rest = net.split_node_full_svd(
      n_block,
      output_order[:2],
      output_order[2:],
      max_singular_values=max_bond_dim,
      max_truncation_err=auto_trunc_max_err)

  trunc_err = tf.norm(s_rest)
  nrm = tf.norm(ns.tensor)
  ns.tensor = tf.divide(ns.tensor, nrm)
  L1_new = ns.tensor

  #output_order = [nL0_i["L"], nu["p_out_1"], es1]
  output_order = [nL0_i["L"], nu[1], ns[0]]
  nG1_new = net.contract(b0)
  nG1_new.reorder_edges(output_order)
  G1_new = nG1_new.tensor

  #output_order = [es2, nvh["p_out_2"], nL2_i["R"]]
  output_order = [ns[1], nvh[1], nL2_i["R"]]
  nG2_new = net.contract(b5)
  nG2_new.reorder_edges(output_order)
  G2_new = nG2_new.tensor

  return G1_new, L1_new, G2_new, nrm, trunc_err


apply_2site_schmidt_canonical_defuned = tf.contrib.eager.defun(
    apply_2site_schmidt_canonical)


def apply_2site_generic(op, A1, A2, max_bond_dim=None, auto_trunc_max_err=0.0):
  """Applies a two-site local operator to an MPS.
    Takes two MPS site tensors and returns new ones, with a center matrix.
    """
  if tf.executing_eagerly():
    # FIXME: Not ideal, but these ops are very costly at compile time
    op_shp = tf.shape(op)
    tf.assert_equal(
        tf.shape(A1)[1],
        op_shp[2],
        message="Operator dimensions do not match MPS physical dimensions.")
    tf.assert_equal(
        tf.shape(A2)[1],
        op_shp[3],
        message="Operator dimensions do not match MPS physical dimensions.")

  net = tn.TensorNetwork()
  nA1 = net.add_node(A1, axis_names=["L", "p", "R"])
  nA2 = net.add_node(A2, axis_names=["L", "p", "R"])
  nop = net.add_node(op, axis_names=["p_out_1", "p_out_2", "p_in_1", "p_in_2"])

  net.connect(nA1["R"], nA2["L"])
  net.connect(nA1["p"], nop["p_in_1"])
  net.connect(nA2["p"], nop["p_in_2"])

  output_order = [nA1["L"], nop["p_out_1"], nop["p_out_2"], nA2["R"]]

  nA12 = net.contract_between(nA1, nA2)
  n_block = net.contract_between(nop, nA12)

  nA1_new, nC, nA2_new, s_rest = net.split_node_full_svd(
      n_block,
      output_order[:2],
      output_order[2:],
      max_singular_values=max_bond_dim,
      max_truncation_err=auto_trunc_max_err)

  trunc_err = tf.norm(s_rest)

  return nA1_new.tensor, nC.tensor, nA2_new.tensor, trunc_err


apply_2site_generic_defuned = tf.contrib.eager.defun(apply_2site_generic)


@tf.contrib.eager.defun
def TMeigs_power_method(tensors,
                        direction,
                        init=None,
                        precision=1E-12,
                        nmax=100000):
  """
    calculate the left and right dominant eigenvector of the MPS-unit-cell transfer operator
    using the power method
    Args:
        tensors (list of tf.Tensor): mps tensors
        direction (int or str): if direction in (1,'l','left')   return the left dominant EV
                                if direction in (-1,'r','right') return the right dominant EV
        init (tf.Tensor):       initial guess for the eigenvector
        precision (float):      desired precision of the dominant eigenvalue
        nmax (int):             max number of iterations      
    Returns:
        eta (tf.Tensor):        the eigenvalue
        x (tf.Tensor):          the dominant eigenvector (in matrix form)
        nit (tf.Tensor):        number of iterations
        diff (tf.Tensor):       final precision
    """

  As = [t for t in tensors]  #won't compile without this
  if not np.all(As[0].dtype == t.dtype for t in As):
    raise TypeError('TMeigs_power_method: all As have to have the same dtype')

  if init:
    x = init
  else:
    x = tf.diag(tf.ones(shape=[As[0].shape[0]], dtype=As[0].dtype))
  if not As[0].dtype == x.dtype:
    raise TypeError('TMeigs_power_method: `init` has other dtype than `As`')

  x /= tf.linalg.norm(x)
  dtype = x.dtype

  def do_step(n, eta, state, diff):
    newstate = transfer_op(As, As, direction, state)
    eta = tf.linalg.norm(newstate)
    newstate /= eta
    diff = tf.cast(tf.linalg.norm(state - newstate), dtype.real_dtype)
    return n + 1, eta, newstate, diff

  def stopping_criterion(n, eta, state, diff):
    return tf.less(tf.cast(precision, dtype.real_dtype), diff)

  def cond(n, eta, state, diff):
    return tf.cond(
        tf.less(0, n), lambda: tf.cond(
            tf.less(n, nmax), lambda: stopping_criterion(n, eta, state, diff),
            lambda: False), lambda: True)

  n_final, eta, state_final, diff = tf.while_loop(
      cond, do_step,
      (0, tf.cast(0.0, dtype), x, tf.cast(1.0, dtype.real_dtype)))
  return eta, state_final, n_final, diff


def TMeigs(tensors,
           direction,
           init=None,
           precision=1E-12,
           ncv=50,
           nmax=1000,
           numeig=1,
           which='LR'):
  """
    calculate the left or right dominant eigenvector of the MPS-unit-cell transfer operator using sparse 
    method.
    Notes: - Currently  only works in eager mode.
           - the implementation uses scipy's sparse module (eigs). tf.Tensor are mapped to numpy arrays and back
             to tf.Tensor for each call to matrix-vector product. This is not optimal and will be fixed at some alter stage
    Args:
        tensors (list of tf.Tensor): mps tensors
        direction (int or str): if direction in (1,'l','left')   return the left dominant EV
                                if direction in (-1,'r','right') return the right dominant EV
        init (tf.Tensor):       initial guess for the eigenvector
        precision (float):      desired precision of the dominant eigenvalue
        ncv(int):               number of Krylov vectors
        nmax (int):             max number of iterations
        numeig (int):           hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                                to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
        which (str):            hyperparameter, passed to scipy.sparse.linalg.eigs; which eigen-vector to target
                                can be ('LM','LA,'SA','LR'), refer to scipy.sparse.linalg.eigs documentation for details
    Returns:
        eta (tf.Tensor):        the eigenvalue
        x (tf.Tensor):          the dominant eigenvector (in matrix form)
    """
  #FIXME: add a tensorflow native eigs

  if not np.all(tensors[0].dtype == t.dtype for t in tensors):
    raise TypeError('TMeigs: all tensors have to have the same dtype')
  dtype = tensors[0].dtype
  Dl = tensors[0].shape[0]
  Dr = tensors[-1].shape[2]
  if tf.executing_eagerly() and Dl != Dr:
    raise ValueError(
        " in TMeigs: left and right ancillary dimensions of the MPS do not match"
    )
  if np.all(init != None):
    initial = init.numpy()

  def mv(vector):
    x = tf.reshape(tf.convert_to_tensor(vector), (Dl, Dl))
    out = transfer_op(tensors, tensors, direction, x).numpy()
    return out.reshape(out.shape[0] * out.shape[1])

  LOP = LinearOperator((Dl * Dl, Dr * Dr),
                       matvec=mv,
                       dtype=dtype.as_numpy_dtype)
  if numeig >= LOP.shape[0] - 1:
    warnings.warn(
        'TMeigs: numeig+1 ({0}) > dimension of transfer operator ({1}) changing value to numeig={2}'
        .format(numeig + 1, LOP.shape[0], LOP.shape[0] - 2))
    while numeig >= (LOP.shape[0] - 1):
      numeig -= 1

  eta, vec = eigs(
      LOP, k=numeig, which=which, v0=init, maxiter=nmax, tol=precision, ncv=ncv)
  m = np.argmax(np.real(eta))
  while np.abs(np.imag(eta[m])) / np.abs(np.real(eta[m])) > 1E-4:
    numeig = numeig + 1
    print(
        'found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'
        .format(numeig))
    eta, vec = eigs(
        LOP,
        k=numeig,
        which=which,
        v0=init,
        maxiter=nmax,
        tol=precision,
        ncv=ncv)
    m = np.argmax(np.real(eta))

  if np.issubdtype(dtype.as_numpy_dtype, np.floating):
    out = np.reshape(vec[:, m], (Dl, Dl))
    if np.linalg.norm(np.imag(out)) > 1E-10:
      raise TypeError(
          "TMeigs: dtype was float, but returned eigenvector had a large imaginary part; something went wrong here!"
      )
    return tf.convert_to_tensor(np.real(eta[m])), tf.convert_to_tensor(
        np.real(out))
  elif np.issubdtype(dtype.as_numpy_dtype, np.complexfloating):
    return tf.convert_to_tensor(eta[m]), tf.reshape(
        tf.convert_to_tensor(vec[:, m]), (Dl, Dl))


def initialize_mps_tensors_numpy(initializer_function,
                                 D,
                                 d,
                                 dtype,
                                 minval=-0.1,
                                 maxval=0.1):
  """
    return a list of numpy tensors initialized with `initializer_function`
  
    Args:
        initializer_function (callable):      an initializer function
                                             this function will be called as 
                                             `initializer_function(shape=[D[n-1], d[n], D[n], dtyper=dtype, *args, **kwargs])`
        D (list of int):                      bond dimensions of the MPS
        d (list of int):                      physical dimensions of the MPS
        dtype (tf dtype):                     dtype of the tensors
    Returns:
        list of np.ndarray:  the mps tensors

    """
  N = len(d)
  Ds = [1]
  for n in range(N):
    if Ds[-1] * d[n] < D:
      Ds.append(Ds[-1] * d[n])
    else:
      Ds.append(D)
  Ds[-1] = 1
  for n in range(N - 1):
    if Ds[N - n] * d[N - 1 - n] < Ds[N - n - 1]:
      Ds[N - 1 - n] = Ds[N - n] * d[n]

  if np.issubdtype(dtype, np.floating):
    return [
        initializer_function([Ds[n], d[n], Ds[n + 1]]).astype(dtype) *
        (maxval - minval) + minval for n in range(len(d))
    ]

  elif np.issubdtype(dtype, np.complexfloating):
    return [
        initializer_function([Ds[n], d[n], Ds[n + 1]]).astype(dtype) *
        (maxval - minval) + minval + 1.0j * initializer_function(
            [Ds[n], d[n], Ds[n + 1]]).astype(dtype) * (maxval - minval) + minval
        for n in range(len(d))
    ]


def initialize_mps_tensors(initializer_function, D, d, dtype, *args, **kwargs):
  """
    return a list of numpy tensors initialized with `initializer_function`
    
      Args:
          initializer_function (callable):      an initializer function
                                               this function will be called as 
                                               `initializer_function(shape=[D[n-1], d[n], D[n], dtyper=dtype, *args, **kwargs])`
          D (list of int):                      bond dimensions of the MPS
          d (list of int):                      physical dimensions of the MPS
          dtype (tf dtype):                     dtype of the tensors
          *args, **kwargs:                      parameters passed to `initializer_function`
      Returns:
          list of tf.Tensor:  the mps tensors
    
    """
  if np.issubdtype(dtype.as_numpy_dtype, np.floating):
    return [
        initializer_function(
            shape=[D[n], d[n], D[n + 1]], dtype=dtype, *args, **kwargs)
        for n in range(len(d))
    ]

  elif np.issubdtype(dtype.as_numpy_dtype, np.complexfloating):
    return [
        tf.complex(
            initializer_function(
                shape=[D[n], d[n], D[n + 1]],
                dtype=dtype.real_dtype,
                *args,
                **kwargs),
            initializer_function(
                shape=[D[n], d[n], D[n + 1]],
                dtype=dtype.real_dtype,
                *args,
                **kwargs)) for n in range(len(d))
    ]


#NOTE: this one can't be @tf.contrib.eager.defun'ed
def restore_helper(tensors,
                   init=None,
                   precision=1E-12,
                   ncv=50,
                   nmax=100000,
                   numeig=1,
                   pinv=1E-30):
  """
    Helper function for putting InfiniteMPSCentralGauge into central form using TMeigs
    Args:
        init (tf.tensor):     initial guess for the eigenvector
        precision (float):    desired precision of the dominant eigenvalue
        ncv (int):            number of Krylov vectors
        nmax (int):           max number of iterations
        numeig (int):         hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                              to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
        pinv (float):         pseudoinverse cutoff
    Returns:
        As (list of tf.Tensors):  the mps matrices
        mat (tf.Tensor):          center matrix
        connector (tf.Tensor):    connector matrix
        right_mat (tf.Tensor):    right boundary matrix
    """
  As = copy.copy(tensors)  #[t for t in tensors] #won't compile without this

  if not np.all(As[0].dtype == t.dtype for t in As):
    raise TypeError('TMeigs_power_method: all As have to have the same dtype')
  dtype = As[0].dtype

  eta, l = TMeigs(
      tensors=As,
      direction='left',
      init=init,
      nmax=nmax,
      precision=precision,
      ncv=ncv,
      numeig=numeig)

  sqrteta = tf.cast(tf.sqrt(tf.real(eta)), dtype)
  As[0] /= sqrteta

  l = l / tf.trace(l)
  l = (l + tf.conj(tf.transpose(l))) / 2.0

  eigvals_left, u_left = tf.linalg.eigh(l)

  eigvals_left /= tf.reduce_sum(eigvals_left, axis=0)
  abseigvals_left = tf.abs(eigvals_left)
  mask = tf.greater(abseigvals_left, pinv)
  eigvals_left = tf.where(mask, eigvals_left,
                          tf.zeros(eigvals_left.shape, dtype=dtype))
  inveigvals_left = tf.where(mask, 1.0 / eigvals_left,
                             tf.zeros(eigvals_left.shape, dtype=dtype))

  y = ncon([u_left, tf.diag(tf.sqrt(eigvals_left))], [[-2, 1], [1, -1]])
  invy = ncon([tf.diag(tf.sqrt(inveigvals_left)),
               tf.conj(u_left)], [[-2, 1], [-1, 1]])

  eta, r = TMeigs(
      tensors=As,
      direction='right',
      init=init,
      nmax=nmax,
      precision=precision,
      ncv=ncv,
      numeig=numeig)

  r = r / tf.trace(r)
  r = (r + tf.conj(tf.transpose(r))) / 2.0

  eigvals_right, u_right = tf.linalg.eigh(r)

  eigvals_right /= tf.reduce_sum(eigvals_right, axis=0)
  abseigvals_right = tf.abs(eigvals_right)
  mask = tf.greater(abseigvals_right, pinv)
  eigvals_right = tf.where(mask, eigvals_right,
                           tf.zeros(eigvals_right.shape, dtype=dtype))
  inveigvals_right = tf.where(mask, 1.0 / eigvals_right,
                              tf.zeros(eigvals_right.shape, dtype=dtype))

  x = ncon([u_right, tf.diag(tf.sqrt(eigvals_right))], [[-1, 1], [1, -2]])
  invx = ncon([tf.diag(tf.sqrt(inveigvals_right)),
               tf.conj(u_right)], [[-1, 1], [-2, 1]])
  lam, U, V = tf.linalg.svd(ncon([y, x], [[-1, 1], [1, -2]]))
  lam = tf.cast(lam, dtype)

  As[0] = ncon(  #absorb everything on the left end 
      [tf.diag(lam), tf.conj(V), invx, As[0]],
      [[-1, 1], [2, 1], [2, 3], [3, -2, -3]])
  As[-1] = ncon([As[-1], invy, U], [[-1, -2, 1], [1, 2], [2, -3]])

  for n in range(len(As) - 1):
    tensor, mat, _ = prepare_tensor_QR(As[n], direction=1)
    As[n] = tensor
    As[n + 1] = ncon([mat, As[n + 1]], [[-1, 1], [1, -2, -3]])

  Z = ncon([As[-1], tf.conj(As[-1])], [[1, 2, 3], [1, 2, 3]]) / tf.cast(
      As[-1].shape[2], dtype)
  As[-1] /= tf.sqrt(Z)
  lam = lam / tf.linalg.norm(lam)
  mat = tf.diag(lam)
  connector = tf.diag(1.0 / lam)
  right_mat = tf.diag(lam)

  return As, mat, connector, right_mat


@tf.contrib.eager.defun
def restore_helper_power_method(tensors,
                                init=None,
                                precision=1E-12,
                                nmax=100000,
                                pinv=1E-30):
  """
    Helper function for putting InfiniteMPSCentralGauge into central form using TMeigs_power_method
    Args:
        init (tf.tensor):     initial guess for the eigenvector
        precision (float):    desired precision of the dominant eigenvalue
        ncv (int):            number of Krylov vectors
        nmax (int):           max number of iterations
        numeig (int):         hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                              to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
        pinv (float):         pseudoinverse cutoff
    Returns:
        As (list of tf.Tensors):  the mps matrices
        mat (tf.Tensor):          center matrix
        connector (tf.Tensor):    connector matrix
        right_mat (tf.Tensor):    right boundary matrix
    """

  As = copy.copy(tensors)  #[t for t in tensors] #won't compile without this
  newAs = []
  if not np.all(As[0].dtype == t.dtype for t in As):
    raise TypeError('TMeigs_power_method: all As have to have the same dtype')
  dtype = As[0].dtype

  if init:
    x = init
  else:
    x = tf.diag(tf.ones(shape=[As[0].shape[0]], dtype=As[0].dtype))
  if not As[0].dtype == x.dtype:
    raise TypeError('TMeigs_power_method: `init` has other dtype than `As`')

  x /= tf.linalg.norm(x)
  dtype = x.dtype

  def do_step_left(
      n,
      eta,
      state,
      diff,
  ):
    newstate = transfer_op(As, As, 'l', state)
    eta = tf.linalg.norm(newstate)
    newstate /= eta
    diff = tf.cast(tf.linalg.norm(state - newstate), dtype.real_dtype)
    return n + 1, eta, newstate, diff

  def do_step_right(
      n,
      eta,
      state,
      diff,
  ):
    newstate = transfer_op(As, As, 'r', state)
    eta = tf.linalg.norm(newstate)
    newstate /= eta
    diff = tf.cast(tf.linalg.norm(state - newstate), dtype.real_dtype)
    return n + 1, eta, newstate, diff

  def stopping_criterion(n, eta, state, diff):
    return tf.less(tf.cast(precision, dtype.real_dtype), diff)

  def cond(n, eta, state, diff):
    return tf.cond(
        tf.less(0, n), lambda: tf.cond(
            tf.less(n, nmax), lambda: stopping_criterion(n, eta, state, diff),
            lambda: False), lambda: True)

  _, eta, l, _ = tf.while_loop(
      cond, do_step_left,
      (0, tf.cast(0.0, dtype), x, tf.cast(1.0, dtype.real_dtype)))
  _, eta, r, _ = tf.while_loop(
      cond, do_step_right,
      (0, tf.cast(0.0, dtype), x, tf.cast(1.0, dtype.real_dtype)))

  sqrteta = tf.cast(tf.sqrt(tf.real(eta)), dtype)
  As[0] /= sqrteta

  l = l / tf.trace(l)
  l = (l + tf.conj(tf.transpose(l))) / 2.0

  eigvals_left, u_left = tf.linalg.eigh(l)

  eigvals_left /= tf.reduce_sum(eigvals_left, axis=0)
  abseigvals_left = tf.abs(eigvals_left)
  mask = tf.greater(abseigvals_left, pinv)
  eigvals_left = tf.where(mask, eigvals_left,
                          tf.zeros(eigvals_left.shape, dtype=dtype))
  inveigvals_left = tf.where(mask, 1.0 / eigvals_left,
                             tf.zeros(eigvals_left.shape, dtype=dtype))

  y = ncon([u_left, tf.diag(tf.sqrt(eigvals_left))], [[-2, 1], [1, -1]])
  invy = ncon([tf.diag(tf.sqrt(inveigvals_left)),
               tf.conj(u_left)], [[-2, 1], [-1, 1]])

  r = r / tf.trace(r)
  r = (r + tf.conj(tf.transpose(r))) / 2.0

  eigvals_right, u_right = tf.linalg.eigh(r)

  eigvals_right /= tf.reduce_sum(eigvals_right, axis=0)
  abseigvals_right = tf.abs(eigvals_right)
  mask = tf.greater(abseigvals_right, pinv)
  eigvals_right = tf.where(mask, eigvals_right,
                           tf.zeros(eigvals_right.shape, dtype=dtype))
  inveigvals_right = tf.where(mask, 1.0 / eigvals_right,
                              tf.zeros(eigvals_right.shape, dtype=dtype))

  x = ncon([u_right, tf.diag(tf.sqrt(eigvals_right))], [[-1, 1], [1, -2]])
  invx = ncon([tf.diag(tf.sqrt(inveigvals_right)),
               tf.conj(u_right)], [[-1, 1], [-2, 1]])
  lam, U, V = tf.linalg.svd(ncon([y, x], [[-1, 1], [1, -2]]))
  lam = tf.cast(lam, dtype)

  As[0] = ncon(  #absorb everything on the left end 
      [tf.diag(lam), tf.conj(V), invx, As[0]],
      [[-1, 1], [2, 1], [2, 3], [3, -2, -3]])
  As[-1] = ncon([As[-1], invy, U], [[-1, -2, 1], [1, 2], [2, -3]])

  for n in range(len(As) - 1):
    tensor, mat, _ = prepare_tensor_QR(As[n], direction=1)
    As[n] = tensor
    As[n + 1] = ncon([mat, As[n + 1]], [[-1, 1], [1, -2, -3]])

  Z = ncon([As[-1], tf.conj(As[-1])], [[1, 2, 3], [1, 2, 3]]) / tf.cast(
      As[-1].shape[2], dtype)
  As[-1] /= tf.sqrt(Z)
  lam = lam / tf.linalg.norm(lam)
  mat = tf.diag(lam)
  connector = tf.diag(1.0 / lam)
  right_mat = tf.diag(lam)
  return As, mat, connector, right_mat


def compile_ncon(on=True):
  global ncon
  if on:
    ncon = ncon_defuned
  else:
    ncon = ncon_tn


def compile_contractions(on=True):
  global transfer_op, add_layer
  if on:
    transfer_op = transfer_op_defuned
    add_layer = add_layer_defuned
  else:
    pass


def compile_decomps(on=True):
  global prepare_tensor_SVD
  global prepare_tensor_QR
  global apply_2site_generic
  global apply_2site_schmidt_canonical
  if on:
    prepare_tensor_QR = prepare_tensor_QR_defuned
    prepare_tensor_SVD = prepare_tensor_SVD_defuned
    apply_2site_generic = apply_2site_generic_defuned
    apply_2site_schmidt_canonical = apply_2site_schmidt_canonical_defuned
  else:
    pass


# Default to defuned
compile_ncon(True)
compile_contractions(True)
compile_decomps(True)
