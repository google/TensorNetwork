Quickstart
========
To get started, let's first install the TensorNetwork library.

.. code-block:: bash

  pip3 install tensornetwork


Nodes
-----
Nodes are one of the basic building blocks of a tensor network. They represent a tensor in the computation. Each axis will have a corresponding edge that can possibly connect different nodes (or even the same node) together. The number of edges represents the order of the underlying tensor. For example, a node without any edges is a scalar, a node with one edge is a vector, etc.

.. code-block:: python3

  import tensornetwork as tn
  import numpy as np 

  node = tn.Node(np.eye(2))
  print(node.tensor)  # This is how you access the underlying tensor.

Edges
-----
Edges describe different contractions of the underlying tensors in the tensor network. The edges are associated to the tensor axes involved in the contraction. There are 3 basic kinds of edges in a tensor network:

**Standard Edges**

Standard edges are like any other edge you would find in an undirected graph. They connect 2 different nodes and represent a dot product between the associated vector spaces. In numpy terms, this edge defines a tensordot operation over the given axes.

**Trace Edges**

Trace edges connect a node to itself. Contracting this type of edge means taking the trace of the matrix formed by the two associated axes.

**Dangling Edge**

Dangling edges are edges that only have one side point to a node, with the other side left “dangling”. These edges can represent output axes, or they can represent intermediate axes that have yet to be connected to other dangling edges. These edges are automatically created when adding a node to the network.

.. code-block:: python3

  a = tn.Node(np.eye(2))
  b = tn.Node(np.eye(2))
  c = tn.Node(np.eye(2))
  # Dangling edges are automatically created at node creation. 
  # We can access them this way.
  dangling_edge = a.get_edge(1)
  # This is the same as above
  dangling_edge = a[1]
  # Create a standard edge by connecting any two separate nodes together.
  # We create a new edge by "connecting" two dangling edges.
  standard_edge = a[0] ^ b[0] # same as tn.connect(a[0], b[0]) 
  # Create a trace edge by connecting a node to itself.
  trace_edge = c[0] ^ c[1]

Connecting Dangling Edges 
-------------------------
One common paradigm in building quantum circuits, for example, is to add computation based on edges rather than nodes. That is, it’s very common to want to connect two dangling edges rather than trying to keep track of node axes.

In this example, we have a single qubit quantum circuit where we apply a Hadamard operation several times. We connect the dangling edge of the qubit to the Hadamard operation and return the resulting “output” edge.

.. code-block:: python3

  def apply_hadamard(edge):
    hadamard_op = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    hadamard_node = tn.Node(hadamard_op)
    # Connect the "qubit edge" to the operator "input edge" 
    edge ^ hadamard_node[0]
    return hadamard_node[1]  # This is the "output edge".

  # Build the quantum circuit.
  qubit = tn.Node(np.array([1.0, 0.0])) # A "zero state" qubit.
  qubit_edge = qubit[0]
  for i in range(5):
    qubit_edge = apply_hadamard(qubit_edge)

Edge Contraction
----------------
Contracting an edge is just a simple call. The tensor network API is smart enough to figure out what type of edge was passed and will do the correct computation accordingly.

This example code calculates the dot product of two vectors.

.. code-block:: python3

  a = tn.Node(np.ones(2))
  b = tn.Node(np.ones(2))
  edge = a[0] ^ b[0]
  c = tn.contract(edge)
  print(c.tensor) # Should print 2.0


Optimized Contractions
----------------------
At intermediate states of a computation, it’s very common for two nodes to have multiple edges connecting them. If only one of those edges is contracted, then all of the remaining edges become trace edges. This is usually very inefficient, as the new node will allocate significantly more memory than is ultimately required. Since trace edges only sum the diagonal of the underlying matrix, all of the other values calculated during the first contraction are useless. It is always more efficient to contract all of these edges simultaneously.

The methods `contract_between` or `contract_parallel` will do this for you automatically. You should see huge speedups when comparing these methods against contracting one edge at a time.

.. code-block:: python3

  def one_edge_at_a_time(a, b):
    node1 = tn.Node(a)
    node2 = tn.Node(b)
    edge1 = node1[0] ^ node2[0]
    edge2 = node1[1] ^ node2[1]
    tn.contract(edge1)
    result = tn.contract(edge2)
    return result.tensor

  def use_contract_between(a, b):
    node1 = tn.Node(a)
    node2 = tn.Node(b)
    node1[0] ^ node2[0]
    node1[1] ^ node2[1]
    # This is the same as
    # tn.contract_between(node1, node2)
    result = node1 @ node2 
    return result.tensor

  a = np.ones((1000, 1000))
  b = np.ones((1000, 1000))

.. code-block:: python3

  >>> print("Running one_edge_at_a_time")
  >>> %timeit one_edge_at_a_time(a, b)
  >>> print("Running use_cotract_between")
  >>> %timeit use_contract_between(a, b)

  # Running one_edge_at_a_time
  # 10 loops, best of 3: 41.8 ms per loop
  # Running use_cotract_between
  # 1000 loops, best of 3: 1.32 ms per loop

--
