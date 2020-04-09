=======================
Basic API Tutorial
=======================

.. code:: 

    !pip install tensornetwork jax jaxlib
    import numpy as np
    import jax
    import tensornetwork as tn


.. parsed-literal::

    Collecting tensornetwork
    [?25l  Downloading https://files.pythonhosted.org/packages/6d/ed/ea8087d21b73650a3df360e5ae58d8b3ac8cb8787493caa32311355dc4ab/tensornetwork-0.2.0-py3-none-any.whl (202kB)
    [K     |████████████████████████████████| 204kB 10.3MB/s 
    [?25hRequirement already satisfied: jax in /usr/local/lib/python3.6/dist-packages (0.1.52)
    Requirement already satisfied: jaxlib in /usr/local/lib/python3.6/dist-packages (0.1.36)
    Requirement already satisfied: numpy>=1.16 in /usr/local/lib/python3.6/dist-packages (from tensornetwork) (1.17.4)
    Collecting graphviz>=0.11.1
      Downloading https://files.pythonhosted.org/packages/f5/74/dbed754c0abd63768d3a7a7b472da35b08ac442cf87d73d5850a6f32391e/graphviz-0.13.2-py2.py3-none-any.whl
    Collecting h5py>=2.9.0
    [?25l  Downloading https://files.pythonhosted.org/packages/60/06/cafdd44889200e5438b897388f3075b52a8ef01f28a17366d91de0fa2d05/h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl (2.9MB)
    [K     |████████████████████████████████| 2.9MB 46.2MB/s 
    [?25hRequirement already satisfied: opt-einsum>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensornetwork) (3.1.0)
    Requirement already satisfied: fastcache in /usr/local/lib/python3.6/dist-packages (from jax) (1.1.0)
    Requirement already satisfied: absl-py in /usr/local/lib/python3.6/dist-packages (from jax) (0.8.1)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from jax) (1.12.0)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from jaxlib) (1.3.3)
    Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.6/dist-packages (from jaxlib) (3.10.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from protobuf>=3.6.0->jaxlib) (42.0.1)
    Installing collected packages: graphviz, h5py, tensornetwork
      Found existing installation: graphviz 0.10.1
        Uninstalling graphviz-0.10.1:
          Successfully uninstalled graphviz-0.10.1
      Found existing installation: h5py 2.8.0
        Uninstalling h5py-2.8.0:
          Successfully uninstalled h5py-2.8.0
    Successfully installed graphviz-0.13.2 h5py-2.10.0 tensornetwork-0.2.0


Section 1: Basic usage.
=======================

In this section, we will go over basic linear algebra operations and how
to create them using a TensorNetwork. While at first it may seem more
complicated to use a tensornetwork rather than just doing the operations
by hand, we will use the skills developed in this section to start
building and contracting very complicated tensor networks that would be
very difficult to do otherwise.

Let’s begin by doing the most basic operation possible, a vector dot.

.. code:: 

    # Next, we add the nodes containing our vectors.
    a = tn.Node(np.ones(10))
    # Either tensorflow tensors or numpy arrays are fine.
    b = tn.Node(np.ones(10))
    # We "connect" these two nodes by their "0th" edges.
    # This line is equal to doing `tn.connect(a[0], b[0])
    # but doing it this way is much shorter.
    edge = a[0] ^ b[0]
    # Finally, we contract the edge, giving us our new node with a tensor
    # equal to the inner product of the two earlier vectors
    c = tn.contract(edge)
    # You can access the underlying tensor of the node via `node.tensor`.
    # To convert a Eager mode tensorflow tensor into 
    print(c.tensor)


.. parsed-literal::

    10.0


Edge-centric connection.
------------------------

When a node is created in the TensorNetwork, that node is automatically
filled with dangling-edges. To connect two nodes together, we actually
remove the two danging edges in the nodes and replace them with a
standard/trace edge.

.. code:: 

    a = tn.Node(np.eye(2))
    # Notice that a[0] is actually an "Edge" type.
    print("The type of a[0] is:", type(a[0]))
    # This is a dangling edge, so this method will 
    print("Is a[0] dangling?:", a[0].is_dangling())


.. parsed-literal::

    The type of a[0] is: <class 'tensornetwork.network_components.Edge'>
    Is a[0] dangling?: True


Now, let’s connect a[0] to a[1]. This will create a “trace” edge.

.. code:: 

    trace_edge = a[0] ^ a[1]
    # Notice now that a[0] and a[1] are actually the same edge.
    print("Are a[0] and a[1] the same edge?:", a[0] is a[1])
    print("Is a[0] dangling?:", a[0].is_dangling())


.. parsed-literal::

    Are a[0] and a[1] the same edge?: True
    Is a[0] dangling?: False


Axis naming.
------------

Sometimes, using the axis number is very inconvient and it can be hard
to keep track of the purpose of certain edges. To make it easier, you
can optionally add a name to each of the axes of your node. Then you can
get the respective edge by indexing using the name instead of the
number.

.. code:: 

    # Here, a[0] is a['alpha'] and a[1] is a['beta']
    a = tn.Node(np.eye(2), axis_names=['alpha', 'beta'])
    edge = a['alpha'] ^ a['beta']
    result = tn.contract(edge)
    print(result.tensor)


.. parsed-literal::

    2.0


Section 2. Advanced Network Contractions
========================================

Avoid trace edges.
------------------

While the TensorNetwork library fully supports trace edges, contraction
time is ALWAYS faster if you avoid creating them. This is because trace
edges only sum the diagonal of the underlying matrix, and the rest of
the values (which is a majorit of the total values) are just garbage.
You both waste compute time and memory by having these useless trace
edges.

The main way we support avoid trace edges is via the ``@`` symbol, which
is an alias to ``tn.contract_between``. Take a look at the speedups!

.. code:: 

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
    print("Running one_edge_at_a_time")
    %timeit one_edge_at_a_time(a, b)
    print("Running use_cotract_between")
    %timeit use_contract_between(a, b)


.. parsed-literal::

    Running one_edge_at_a_time
    1 loop, best of 3: 120 ms per loop
    Running use_cotract_between
    10 loops, best of 3: 54.4 ms per loop


We also have ``contract_parallel`` which does the same thing as
``contract_between``, only you pass a single edge instead of two nodes.
This will contract all of the edges “parallel” to the given edge
(meaning all of the edges that share the same two nodes as the given
edge).

Using either method is fine and they will do the exact same thing. In
fact, if you look at the source code, ``contract_parallel`` actually
just calls ``contract_between``. :)

.. code:: 

    def use_contract_parallel(a, b):
      node1 = tn.Node(a)
      node2 = tn.Node(b)
      edge = node1[0] ^ node2[0]
      node1[1] ^ node2[1]
      result = tn.contract_parallel(edge)
      # You can use `get_final_node` to make sure your network 
      # is fully contracted.
      return result.tensor
    
    print("Running contract_parallel")
    %timeit use_contract_parallel(a, b)


.. parsed-literal::

    Running contract_parallel
    10 loops, best of 3: 53.1 ms per loop


Complex Contraction.
--------------------

Remember this crazy hard to write tensor contraction? Well, we’re gonna
do it in about 13 lines of simple code.

.. code:: 

    # Here, we will contract the following shaped network.
    # O - O
    # | X |
    # O - O
    a = tn.Node(np.ones((2, 2, 2)))
    b = tn.Node(np.ones((2, 2, 2)))
    c = tn.Node(np.ones((2, 2, 2)))
    d = tn.Node(np.ones((2, 2, 2)))
    # Make the network fully connected.
    a[0] ^ b[0]
    a[1] ^ c[1]
    a[2] ^ d[2]
    b[1] ^ d[1]
    b[2] ^ c[2]
    c[0] ^ d[0]
    # We are using the "greedy" contraction algorithm.
    # Other algorithms we support include "optimal" and "branch".
    
    # Finding the optimial contraction order in the general case is NP-Hard,
    # so there is no single algorithm that will work for every tensor network.
    # However, there are certain kinds of networks that have nice properties that
    # we can expliot to making finding a good contraction order easier.
    # These types of contraction algorithms are in developement, and we welcome 
    # PRs!
    
    # `tn.reachable` will do a BFS to get all of the nodes reachable from a given
    # node or set of nodes.
    # nodes = {a, b, c, d}
    nodes = tn.reachable(a)
    result = tn.contractors.greedy(nodes)
    print(result.tensor)



.. parsed-literal::

    64.0


.. code:: 

    # To make connecting a network a little less verbose, we have included
    # the NCon API aswell.
    
    # This example is the same as above.
    ones = np.ones((2, 2, 2))
    tn.ncon([ones, ones, ones, ones], 
            [[1, 2, 4], 
             [1, 3, 5], 
             [2, 3, 6],
             [4, 5, 6]])


.. parsed-literal::

    /usr/local/lib/python3.6/dist-packages/tensornetwork/ncon_interface.py:130: UserWarning: Suboptimal ordering detected. Edges ['4'] are not adjacent in the contraction order to edges ['2'], connecting nodes ['con(tensor_0,tensor_1)', 'tensor_2']. Deviating from the specified ordering!
      list(map(str, nodes_to_contract))))




.. parsed-literal::

    array(64.)



.. code:: 

    # To specify dangling edges, simply use a negative number on that index.
    
    ones = np.ones((2, 2))
    tn.ncon([ones, ones], [[-1, 1], [1, -2]])




.. parsed-literal::

    array([[2., 2.],
           [2., 2.]])



Section 3: Node splitting.
==========================

In the final part of this colab, will go over the SVD node splitting
methods.

.. code:: 

    # To make the singular values very apparent, we will just take the SVD of a
    # diagonal matrix.
    diagonal_array = np.array([[2.0, 0.0, 0.0],
                               [0.0, 2.5, 0.0],
                               [0.0, 0.0, 1.5]]) 

.. code:: 

    # First, we will go over the simple split_node method.
    a = tn.Node(diagonal_array)
    u, vh, _ = tn.split_node(
        a, left_edges=[a[0]], right_edges=[a[1]])
    print("U node")
    print(u.tensor)
    print()
    print("V* node")
    print(vh.tensor)



.. parsed-literal::

    U node
    [[0.        1.4142135 0.       ]
     [1.5811388 0.        0.       ]
     [0.        0.        1.2247449]]
    
    V* node
    [[0.        1.5811388 0.       ]
     [1.4142135 0.        0.       ]
     [0.        0.        1.2247449]]


.. code:: 

    # Now, we can contract u and vh to get back our original tensor!
    print("Contraction of U and V*:")
    print((u @ vh).tensor)


.. parsed-literal::

    Contraction of U and V*:
    [[1.9999999 0.        0.       ]
     [0.        2.5       0.       ]
     [0.        0.        1.5000001]]


We can also drop the lowest singular values in 2 ways, 1. By setting
``max_singular_values``. This is the maximum number of the original
singular values that we want to keep. 2. By setting ``max_trun_error``.
This is the maximum amount the sum of the removed singular values can
be.

.. code:: 

    # We can also drop the lowest singular values in 2 ways, 
    # 1. By setting max_singular_values. This is the maximum number of the original
    # singular values that we want to keep.
    a = tn.Node(diagonal_array)
    u, vh, truncation_error = tn.split_node(
        a, left_edges=[a[0]], right_edges=[a[1]], max_singular_values=2)
    # Notice how the two largest singular values (2.0 and 2.5) remain
    # but the smallest singular value (1.5) is removed.
    print((u @ vh).tensor)


.. parsed-literal::

    [[1.9999999 0.        0.       ]
     [0.        2.5       0.       ]
     [0.        0.        0.       ]]


We can see the values of the removed singular values by looking at the
returned ``truncation_error``

.. code:: 

    # truncation_error is just a normal tensorflow tensor.
    print(truncation_error)


.. parsed-literal::

    [1.5]


Section 4: running on GPUs
==========================

To get this running on a GPU, we recommend using the JAX backend, as it
has nearly the exact same API as numpy.

To get a GPU, go to Runtime -> Change runtime type -> GPU

.. code:: 

    def calculate_abc_trace(a, b, c):
      an = tn.Node(a)
      bn = tn.Node(b)
      cn = tn.Node(c)
      an[1] ^ bn[0]
      bn[1] ^ cn[0]
      cn[1] ^ an[0]
      return (an @ bn @ cn).tensor
    
    a = np.ones((4096, 4096))
    b = np.ones((4096, 4096))
    c = np.ones((4096, 4096))
    
    tn.set_default_backend("numpy")
    print("Numpy Backend")
    %timeit calculate_abc_trace(a, b, c)
    tn.set_default_backend("jax")
    # Running with a GPU: 202 ms
    # Running with a CPU: 2960 ms
    print("JAX Backend")
    %timeit np.array(calculate_abc_trace(a, b, c))


.. parsed-literal::

    Numpy Backend
    1 loop, best of 3: 5.46 s per loop
    JAX Backend


.. parsed-literal::

    /usr/local/lib/python3.6/dist-packages/jax/lib/xla_bridge.py:115: UserWarning: No GPU/TPU found, falling back to CPU.
      warnings.warn('No GPU/TPU found, falling back to CPU.')


.. parsed-literal::

    1 loop, best of 3: 2.96 s per loop


