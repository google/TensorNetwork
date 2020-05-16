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
  a = FunctionalNode(a_val)
  b = FunctionalNode(b_val)
  # Order is initialized on first node(...) call.
  c = a["a,b"] @ b["b,c"]
  # C is lazily evaluated until a c(...), c[...] or c.tensor call.
  np.testing.assert_allclose(c("a", "c").tensor, expected)


def test_trace_edges():
  a_val = np.random.randn(2, 4, 4, 5)
  b_val = np.random.randn(2, 3, 5)
  expected = np.einsum("abbc,adc->d", a_val, b_val)
  a = FunctionalNode(a_val)
  b = FunctionalNode(b_val)
  # Brackets [...] are used for comma separated keys.
  # Parenths (...) are used for arbitrary hasable objects
  c = a["alpha,b,b,c"] @ b("alpha", 5, "c")
  np.testing.assert_allclose(expected, c(5).tensor)

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
  c = b @ a
  d = FunctionalNode(d_val)
  # Nodes can always be reused since they are immutable
  # up to lazy evaluation.
  e = a @ d("b", "d")
  np.testing.assert_allclose(e("a", "d").tensor, expected)
  # You can always redefined the axes of a FunctionalNode
  # You may consider doing this for debugging/documentation reasons.
  f = a("b", "a") @ d("b", "d")
  np.testing.assert_allclose(f("a", "d").tensor, expected)

def test_reuse_node_different_axes():
  a_val = np.random.randn(2, 3)
  b_val = np.random.randn(3, 4)
  expected = a_val @ b_val
  a = FunctionalNode(a_val)
  b = FunctionalNode(b_val)
  c = a("a", "b") @ b("b", "c")
  np.testing.assert_allclose(c("a", "c").tensor, expected)
  # If a FunctionalNode doesn't have predefined axes,
  # you can reuse it several times with different names.
  f = a("x", "y") @ b("y", "z")
  np.testing.assert_allclose(f("x", "z").tensor, expected)

def test_reuse_with_self():
  a_val = np.random.randn(2, 3)
  expected = np.trace(a_val @ a_val.T)
  a = FunctionalNode(a_val, ["a", "b"])
  c = a @ a
  np.testing.assert_allclose(c().tensor, expected)

def test_conj():
  a = FunctionalNode(np.array([1.j, 0.0]), ["a"])
  c = a @ a
  np.testing.assert_allclose(c.tensor, -1.0)
  c = a @ a.conj()
  np.testing.assert_allclose(c.tensor, 1.0)

def test_materialize():
  a_val = np.random.randn(2, 3)
  b_val = np.random.randn(2, 3)
  a = FunctionalNode(a_val, ["a", "b"])
  b = FunctionalNode(b_val, ["d", "b"])
  c = a @ a
  assert len(c.lazy_network.nodes) == 2
  c = c.materialize()
  assert len(c.lazy_network.nodes) == 1

def test_materialize():
  a_val = np.random.randn(2, 3)
  b_val = np.random.randn(2, 3)
  a = FunctionalNode(a_val, ["a", "b"])
  b = FunctionalNode(b_val, ["d", "b"])
  c = a @ a
  assert len(c.lazy_network.nodes) == 2
  c = c.materialize()
  assert len(c.lazy_network.nodes) == 1
