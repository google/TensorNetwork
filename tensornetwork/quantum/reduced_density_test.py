from tensornetwork.quantum import reduced_density
import tensornetwork
import numpy as np
import pytest


CNOT = np.array(
  [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0]  
  ], dtype=np.complex64).reshape((2, 2, 2, 2))

Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex64)

def test_unitary_identity(backend):
  if backend == "pytorch":
    pytest.skip("pytorch doesn't support complex numbers")
  zero_state = np.array([1.0, 0.0], dtype=np.complex64)
  net = tensornetwork.TensorNetwork(backend)
  q1 = net.add_node(zero_state)
  q2 = net.add_node(zero_state)
  y_node = net.add_node(Y)
  q1[0] ^ y_node[0]
  cnot_op = net.add_node(CNOT)
  y_node[1] ^ cnot_op[0]
  q2[0] ^ cnot_op[1]

  _, conj_edges = reduced_density.reduced_density_matrix(net, [cnot_op[3]])
  result = tensornetwork.contractors.optimal(
    net, [cnot_op[2], conj_edges[cnot_op[2]]]).get_final_node()
  np.testing.assert_allclose(result.tensor, 1.0)

