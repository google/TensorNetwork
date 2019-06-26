Tutorial
========
To get started, let's first install the TensorNetwork library.

.. code-block:: bash

  pip3 install tensornetwork


Nodes
-----
Nodes are one of the basic building blocks of a tensor network. They represent a tensor in the computation. Each axis will have a corresponding edge that can possibly connect different nodes (or even the same node) together. The number of edges represents the order of the underlying tensor. For example, a node without any edges is a scalar, a node with one edge is a vector, etc.

.. code-block:: python

  import tensornetwork
  import tensorflow as tf

  net = tensornetwork.TensorNetwork()
  node = net.add_node(tf.eye(2)) # NumPy arrays can also be used.
  print(node.tensor)  # This is how you access the underlying tensor.

Edges
-----
Edges describe different computations of the underlying tensors in the tensor network. Each edge points to which axes of the tensors to do the computation. There are 3 basic kinds of edges in a tensor network:

**Standard Edges**

Standard edges are like any other edge you would find in an undirected graph. They connect 2 different nodes and define a dot product among the given vector spaces. In numpy terms, this edge defines a tensordot operation over the given axes.

**Trace Edges**

Trace edges connect a node to itself. To contract this type of edge, you take a trace of the matrix created by the two given axes.

**Dangling Edge**

Dangling edges are edges that only have one side point to a node, where as the other side is left “dangling”. These edges represent output axes or intermediate axes that have yet to be connected to other dangling edges. These edges are automatically created when adding a node to the network.

.. code-block:: python

  net = tensornetwork.TensorNetwork()
  a = net.add_node(tf.eye(2))
  b = net.add_node(tf.eye(2))
  c = net.add_node(tf.eye(2))
  # Dangling edges are automatically created at node creation. 
  # We can access them this way.
  dangling_edge = a.get_edge(1)
  # This is the same as above
  dangling_edge = a[1]
  # Create a standard edge by connecting any two separate nodes together.
  # We create a new edge by "connecting" two dangling edges.
  standard_edge = net.connect(a[0], b[0]) 
  # Create a trace edge by connecting a node to itself.
  trace_edge = net.connect(c[0], c[1])

Connecting Dangling Edges 
-------------------------
One common paradigm in building quantum circuits is to add computation based on edges rather than nodes. That is, it’s very common to want to connect two dangling edges rather than trying to keep track of node axes.

In this example, we have a single qubit quantum circuit where we apply a Hadamard operation several times. We connect the dangling edge of the qubit to the Hadamard operation and return the resulting “output” edge.

.. code-block:: python

  def apply_hadamard(net, edge):
    hadamard_op = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    hadamard_node = net.add_node(hadamard_op)
    # Connect the "qubit edge" to the operator "input edge" 
    net.connect(edge, hadamard_node[0])
    return hadamard_node[1]  # This is the "output edge".

  # Build the quantum circuit.
  net = tensornetwork.TensorNetwork()
  qubit = net.add_node(np.array([1.0, 0.0])) # A "zero state" qubit.
  qubit_edge = qubit.get_edge(0) # qubit[0] is equivalent.
  for i in range(5):
    qubit_edge = apply_hadamard(net, qubit_edge)

Edge Contraction
----------------
Contracting an edge is just a simple call. The tensor network API is smart enough to figure out what type of edge was passed and will do the correct computation accordingly.

This example code calculates the dot product of two vectors.

.. code-block:: python

  net = tensornetwork.TensorNetwork()
  a = net.add_node(tf.ones(2))
  b = net.add_node(tf.ones(2))
  edge = net.connect(a[0], b[0])
  c = net.contract(edge)
  print(c.tensor.numpy()) # Should print 2.0


Optimized Contractions
----------------------
During computation, it’s very common for two nodes to have multiple edges connecting each other. If only one of the edges are contracted at a time, then all of the remaining edges become trace edges. This is usually very bad for computation, as the new node will allocate significantly more memory than required. Also, since trace edges only sum the diagonal of the underlying matrix, all of the other values calculated during the first contraction are useless. During contraction, it always more efficent to contract all of these edges at the same time.

Doing either ``contract_between`` or ``contract_parallel`` will do this for you automatically. You should see huge speedups when comparing these methods against contracting one edge at a time.

.. code-block:: python

  def one_edge_at_a_time(a, b):
    net = tensornetwork.TensorNetwork()
    node1 = net.add_node(a)
    node2 = net.add_node(b)
    edge1 = net.connect(node1[0], node2[0])
    edge2 = net.connect(node1[1], node2[1])
    net.contract(edge1)
    net.contract(edge2)
    # You can use `get_final_node` to make sure your network 
    # is fully contracted.
    return net.get_final_node().tensor.numpy()

  def use_contract_between(a, b):
    net = tensornetwork.TensorNetwork()
    node1 = net.add_node(a)
    node2 = net.add_node(b)
    net.connect(node1[0], node2[0])
    net.connect(node1[1], node2[1])
    net.contract_between(node1, node2)
    # You can use `get_final_node` to make sure your network 
    # is fully contracted.
    return net.get_final_node().tensor.numpy()

  a = np.ones((1000, 1000))
  b = np.ones((1000, 1000))
  
  >>> print("Running one_edge_at_a_time")
  >>> %timeit one_edge_at_a_time(a, b)
  >>> print("Running use_cotract_between")
  >>> %timeit use_contract_between(a, b)
  # Running one_edge_at_a_time
  # 10 loops, best of 3: 41.8 ms per loop
  # Running use_cotract_between
  # 1000 loops, best of 3: 1.32 ms per loop


Finally, we also have aliased the ``@`` operator to do the same thing as ``contract_between``.

.. code-block:: python3

  # This is the same as net.contract_between(node1, node2)
  node3 = node1 @ node2

