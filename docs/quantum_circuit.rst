Simulating a Quantum Circuit
============================
Quantum circuit simulations is one of the main use cases for tensors networks. This is because all quantum circuit diagrams exactly map to tensor networks! After the tensor network is fully contracted, the resulting tensor will be equal to the wavefunction of the quantum computer just before measurement.

Gates and States
----------------

To get started, lets create a `|0>` state by using 1 node.

.. code-block:: python3

  import tensornetwork as tn
  import numpy as np

  state = tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j]))

One of the important things to realize is that qubits aren't actually represented by :code:`Nodes`,
they are represented by :code:`Edges`!

.. code-block:: python3

  qubit = state[0]

Applying a gate to this qubit is then the same as connecting the dangling edge representing
the qubit to a node representing the gate.

.. code-block:: python3

  # This node represents the Hadamard gate we wish to perform
  # on this qubit.
  hadamard = tn.Node(np.array([[1, 1], [1, -1]])) / np.sqrt(2)
  tn.connect(qubit, hadamard[0]) # Equal to qubit ^ hadamard[0]
  # The "output edge" of the operation represents the qubit after
  #  applying the operation.
  qubit = hadamard[1]
  # Contraction is how you actually "apply" the gate.
  state = state @ hadamard
  print(state.tensor) # array([0.707+0.j, 0.707+0.j])



Multiple Qubits
----------------
Multiple qubits is the same story, just instead of starting with a single node for the state, we can start with a product state instead.

Here, we create an initial `|00>` state and evolve to a `|00> + |11>` bell state.

.. code-block:: python3

  def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qubits):
      tn.connect(qubit_edges[bit], op[i])
      qubit_edges[bit] = op[i + len(operating_qubits)]

  # These are just numpy arrays of the operators.
  H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
  CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
  CNOT[0][0][0][0] = 1
  CNOT[0][1][0][1] = 1
  CNOT[1][0][1][1] = 1
  CNOT[1][1][1][0] = 1
  all_nodes = []
  # NodeCollection allows us to store all of the nodes created under this context.
  with tn.NodeCollection(all_nodes):
    state_nodes = [
        tn.Node(np.array([1.0 + 0.0j, 0.0 + 0.0j],)) for _ in range(2)
    ]
    qubits = [node[0] for node in state_nodes]
    apply_gate(qubits, H, [0])
    apply_gate(qubits, CNOT, [0, 1])
  # We can contract the entire tensornetwork easily with a contractor algorithm
  result = tn.contractors.optimal(
      all_nodes, output_edge_order=qubits)
  print(result.tensor) # array([0.707+0.j, 0.0+0.j], [0.0+0.j, 0.707+0.j])

--
