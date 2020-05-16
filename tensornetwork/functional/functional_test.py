import numpy as np
from tensornetwork.functional import FunctionalNode, FunctionalEdge
import tensornetwork as tn

H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
CNOT = np.array(
  [[1.0, 0.0, 0.0, 0.0],
   [0.0, 1.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 1.0],
   [0.0, 0.0, 1.0, 0.0]]).reshape((2,2,2,2))

def test_sanity_check():
  a_val = np.random.randn(2, 3)
  b_val = np.random.randn(3, 4)
  expected = a_val @ b_val
  a = FunctionalNode(a_val, ["a", "b"])
  b = FunctionalNode(b_val, ["b", "c"])
  c = a @ b
  np.testing.assert_allclose(c("a", "c").tensor, expected)

def test_sanity_check():
  a_val = np.random.randn(2, 3)
  b_val = np.random.randn(3, 4)
  expected = a_val @ b_val
  a = FunctionalNode(a_val, ["a", "b"])
  b = FunctionalNode(b_val, ["b", "c"])
  c = a @ b
  np.testing.assert_allclose(c("a", "c").tensor, expected)

def test_qubits():
  def apply_gate(state, operator, operating_qubits):
    # tn.FunctionalEdge() can be replaced with any hashable object.
    new_edges = [FunctionalEdge() for _ in operating_qubits]
    state = state @ FunctionalNode(operator, operating_qubits + new_edges)
    return (state,) + tuple(new_edges)

  qubits = [FunctionalEdge(), FunctionalEdge()]
  state = FunctionalNode(np.array([[1.0, 0.0], [0.0, 0.0]]), qubits)
  #### Create this circuit
  # |0>-- H -- o --
  #            | 
  # |0> -------X---
  # H gate.
  state, qubits[0] = apply_gate(state, H, [qubits[0]])
  # CNOT gate.
  state, qubits[0], qubits[1] = apply_gate(state, CNOT, [qubits[0], qubits[1]])
  expected = np.array([[1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2.0)
  np.testing.assert_allclose(expected, state.tensor)

def test_reuse_node():
  a_val = np.random.randn(2, 3)
  b_val = np.random.randn(3, 4)
  d_val = np.random.randn(3, 5)
  expected = a_val @ d_val
  a = FunctionalNode(a_val, ["a", "b"])
  b = FunctionalNode(b_val, ["b", "c"])
  # Order no longer matters!
  c = a("a", "b") @ b("c", "b")
  d = FunctionalNode(d_val, ["b", "d"])
  # Nodes can always be reused
  e = a("b", "a") @ d("d", "b")
  np.testing.assert_allclose(e("a", "d").tensor, expected)