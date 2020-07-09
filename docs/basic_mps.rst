Basic Operations with Matrix Product States
===========================================


In this tutorial we will set up a basic Matrix Product state with the use of the TensorNetwork library. We apply two very simple algorithms: retreving a selected component from the structure and calculating the inner product of two MPS.

We'll be using the following libraries:

.. code-block:: python3

  import tensornetwork as tn
  import numpy as np
  import matplotlib.pyplot as plt

The cost of high-dimensional tensors
------------------------------------

We begin by observing the scaling of the cost of storing larger and larger tensors and retrieving a component, as the dimension increases. While the computational complexity of accessing an item in a multidimensional array is :math:`O(1)`, the main cost is the exponentially growing *memory* required to store the tensor.

Let our tensor be :math:`T^{s_1 \cdots s_{\textsf{N}}}` , where each :math:`s_i \in \{1, \ldots d_i\}` and where :math:`d_i` is called a **physical dimension**. Here :math:`\textsf{N}` --called the **Rank** of the tensor-- usually represents the *system size*. For example, in many physical applications, :math:`d_i = 2 \hspace{5pt} \forall i`, and each :math:`s_i` is called a **qubit**.

.. figure:: _static/basic_mps/tensor_1.png
  :align: center

Now let's create a tensor with random entries, with ranks and physical dimensions that run over small ranges:

.. code-block:: python3

  def create_tensor(dimension, rank):
    '''Constructs a tensor of a given rank with random integers'''
    dim_list = tuple([dimension for _ in range(rank)])
    return np.random.random_sample(dim_list)

  ranks = range(2,6)
  dimensions = range(2,40)

  for rank in ranks:
    memory = []
    for dim in dimensions:
        tensor = create_tensor(dim, rank)
        component = tuple(np.random.randint(0, dim, rank))
        memory.append(np.sum([x.nbytes for x in tensor]))
        data = tensor[component]
    plt.loglog(dimensions, memory,'o',ls=':', label = f'N = {rank}')

  plt.legend()
  plt.xlabel('Dimension')
  plt.ylabel('Memory cost')
  plt.show()

This produces the following output:

.. figure:: _static/basic_mps/mps_basic_1.png
  :align: center

We see that the memory required to store this array scales as :math:`\sim d^{\textsf{N}}`. This is an *exponential* growth, which quickly saturates our computational resources.

Introducing Matrix Product States
----------------------------------

One way to work around this "dimensionality catastrophe" is to focus on a particular kind of tensors: Those that can be written as a **Matrix Product State** (the word *state* here is related to the quantum state formed from the coefficients of the tensor):

.. figure:: _static/basic_mps/tensor_2.png
  :align: center

.. math::

  T^{s_1 \ldots s_\textsf{N}}
  =
  \sum_{\{\alpha\}}
  A^{s_1}_{\alpha_1} A^{s_2}_{\alpha_1 \alpha_2} A^{s_3}_{\alpha_2 \alpha_3}  \cdots A^{s_{\textsf{N}-1}}_{\alpha_{\textsf{N}-2}\alpha_{\textsf{N}-1}} A^{s_\textsf{N}}_{\alpha_{\textsf{N}-1}}

where :math:`\{ \alpha \} = \{ \alpha_1, \ldots, \alpha_{\textsf{N}-1}\}`, and :math:`\alpha_i \in \{1 \cdots D_i \}`.

.. figure:: _static/basic_mps/tensor_3.png
  :align: center

The figure above shows the graphical representation of each block of the MPS. The width of each leg represents the fact that each dimension can be different (their labels are in gray). The indices of each physical and bond dimension at each site :math:`j` are labelled :math:`s_j,\alpha_j` respectively. If an edge links two matrices, we say it is connected and a matrix product is understood.

In this tutorial we will take all :math:`D_i` (also called **bond dimension**) equal to a single :math:`D` . Any tensor can be written as a MPS by using **Singular Value Decomposition** (although at the cost of very high bond dimensions --exponentially high as :math:`N\to \infty`)

We begin by creating directly the node structure of the MPS. First we define functions to build each block of the MPS and then the MPS itself:

.. code-block:: python3

  # Retrieving a component

  def block(*dimensions):
    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)

  def create_MPS(rank, dimension, bond_dimension):
    '''Build the MPS tensor'''
    mps = [
        tn.Node( block(dim, bond_dim) )] +
        [tn.Node( block(bond_dim, dim, bond_dim)) for _ in range(rank-2)] +
        [tn.Node( block(bond_dim, dim) )
        ]

    #connect edges to build mps
    connected_edges=[]
    conn=mps[0][1]^mps[1][0]
    for k in range(1,rank-1):
        conn=mps[k][2]^mps[k+1][0]
        connected_edges.append(conn)

    return mps, connected_edges

We will calculate the memory size of MPS of different dimensions and ranks (notice we are able to go much farther than before)

.. code-block:: python3

  dimensions = range(2,9,2)
  MPS_ranks = range(2,150)
  MPS_memory = []

  for dim in dimensions:
      bond_dim = 2
      MPS_memory = []
      for i in range(len(MPS_ranks)):
          rank = MPS_ranks[i]

          # Creating the MPS state:
          ##################################################################
          mps_nodes, mps_edges = create_MPS(rank, dim, bond_dim)
          MPS_memory.append(np.sum([x.tensor.nbytes for x in mps_nodes]))

      # Plot Results
      plt.loglog(MPS_ranks, MPS_memory, 'o',ls=':', label = f'd = {dim}')

  plt.legend()
  plt.xlabel('Tensor Rank')
  plt.ylabel('MPS memory')

  plt.show()

We obtain the following results:

.. figure:: _static/basic_mps/mps_basic_2.png
  :align: center

We see that memory requirements drop significantly: the scaling is now :math:`\sim \textsf{N}^{\textsf{const.}}` (which is polynomial). We can probe higher physical dimensions with less memory.

Retrieving components of a MPS
------------------------------

Let us now retrieve a component of a system of physical dimension 2 and rank :math:`N=20`. This is equivalent to quickly accessing the components of some wavefunction of a 1D quantum chain of 20 qubits! The main computational cost will be the contraction of the MPS bonds. Here we use a simple algorithm to perform the calculation: contract each bond successively until the entire MPS has collapsed to the desired component of the tensor.

With this scheme one can calculate a component of the tensor in a time linear in :math:`N`.

.. code-block:: python3

  ########################################################################
  #----- Retrieving a Component from an MPS by Contracting its edges-----#
  ########################################################################
  dim = 2
  bond_dim = 2
  rank = 20
  components=tuple(np.random.randint(0, dim, rank)) #select randomly the components that we will retrieve
  print(f'components are: {components}')

  mps_nodes, mps_edges = create_MPS(rank, dim, bond_dim)
  for k in range(len(mps_edges)):
      A = tn.contract(mps_edges[k])

  #the last node now has all the edges corresponding to the tensor components.

  print(f'coefficient of the tensor at the selected components: {A.tensor[components]}')

Using the TensorNetwork library
--------------------------------

Now let's use the optimized built-in class from TensorNetwork. First we define a function that gives the byte cost of a given node in our tensor network:

.. code-block:: python3

  #from tensornetwork import matrixproductstates as mps
  # Actually tn initializes with the FiniteMPS class directly!

  dimensions = range(2,9,2)
  ranks = range(2,250)

  bond_dim = 2 # constant
  for phys_dim in dimensions:
      memory = []
      for rank in ranks:
          mps = tn.FiniteMPS.random(
                  d = [phys_dim for _ in range(rank)],
                  D = [bond_dim for _ in range(rank-1)],
                  dtype = np.float32
                  )
          memory.append(np.sum([x.nbytes for x in mps.tensors]))
      plt.loglog(ranks, memory, 'o',ls=':', label = f'd={phys_dim}')
  plt.legend()
  plt.xlabel('Tensor Rank')
  plt.ylabel('MPS Memory')

  plt.plot(MPS_ranks, MPS_memory)
  plt.show()

We obtain:

.. figure:: _static/basic_mps/mps_basic_3.png
  :align: center

Here we show also the last line of the previous plot, which shows the improvements of the optimized class of the library.

Retrieving a Component is now simple: Just contract over each connected edge
and evaluate in the desired component. We'll write the entire algorithm for :math:`\textsf{N} = 24` and :math:`d = d_i = 2` (again to make reference to spins):

.. code-block:: python3

  #Component Retrieval Algorithm:

  rank = 24
  phys_dim = 2
  bond_dim = 6

  # build the mps:
  # Recall the state is canonically normalized when we define the class FiniteMPS
  mpstate = tn.FiniteMPS.random(
    d = [phys_dim for _ in range(rank)],
    D = [bond_dim for _ in range(rank-1)],
    dtype = np.float32
    )

  # connect the edges in the mps and contract over bond dimensions
  nodes = [tn.Node(tensor,f'block_{i}') for i,tensor in enumerate(mpstate.tensors)]

  connected_bonds = [nodes[k].edges[2] ^ nodes[k+1].edges[0] for k in range(-1,rank-1)]

  for x in connected_bonds:
   contracted_node = tn.contract(x) # update for each contracted bond

  # evaluate at a desired component
  component = tuple(np.random.randint(0,phys_dim,rank))

  print(f'Selected components of tensor: {component}')
  print(f'Corresponding coefficient of tensor: {contracted_node.tensor[component]}')

Notice that the implementation still takes a lot of memory resources because we are beginning with a tensor of random components (so our tensor has no symmetries a priori) and then performing contraction on each of the edges.

Even so, we take small bond_dimensions (this could be understood as analyzing *low entanglement* systems). This is not what a typical ground state of a quantum many-body state would look like, specially if it has nearest-neighbor interactions or some special symmetry. In that case the matrices of the MPS will be more sparse and more clever contraction algorithms can be used, besides the fact that we are usually interested in some expectation value rather than exact coefficients. This involves calculating the inner product of two states, which we will show now.

Inner Product of MPS
--------------------

Inner products appear all the time in calculations of expectation values and norms of quantum states. They are sometimes called *overlaps*. Notice that the MPS structure makes the inner product of tensors graphically intuitive, involving the contraction of all the connected edges and bonds:

.. figure:: _static/basic_mps/tensor_4.png
  :align: center

An efficient algorithm takes advantage of the factorization properties of the resulting matrices once the tensors have been put into an MPS form. We make the contractions in a "edge-bond-bond" sequence, sweeping along the graph:

.. code-block:: python3

  np.random.seed(3) # fix seed to build the same tensors each time random is called

  phys_dim = 2
  bond_dim = 2
  ranks = range(2,60)

  for phys_dim in range(2,11,2): # check how physical dim changes scaling
    overlap = []
    for rank in ranks:

        mpstateA = tn.FiniteMPS.random(d = [phys_dim for _ in range(rank)], D = [bond_dim for _ in range(rank-1)], dtype = np.complex128)
        mpstateB = tn.FiniteMPS.random(d = [phys_dim for _ in range(rank)], D = [bond_dim for _ in range(rank-1)], dtype = np.complex128)
        # mpstateB = mpstateA # Check that the random MPS are indeed normalized

        nodesA = [tn.Node(np.conj(tensor),f'A{i}') for i,tensor in enumerate(mpstateA.tensors)]
        nodesB = [tn.Node(tensor,f'B{i}') for i,tensor in enumerate(mpstateB.tensors)]

        connected_bondsA = [nodesA[k].edges[2] ^ nodesA[k+1].edges[0] for k in range(-1,rank-1)]
        connected_bondsB = [nodesB[k].edges[2] ^ nodesB[k+1].edges[0] for k in range(-1,rank-1)]
        connected_edges = [nodesA[k].edges[1] ^ nodesB[k].edges[1] for k in range(rank)]

        for i in range(len(connected_bondsA)):
            contraction = tn.contract(connected_edges[i])
            contraction = tn.contract(connected_bondsA[i])
            contraction = tn.contract(connected_bondsB[i])

        overlap.append(np.abs(contraction.tensor))

    plt.loglog(ranks,overlap,'o',ls=':')
  plt.show()

.. figure:: _static/basic_mps/mps_basic_4.png
  :align: center

Notice how the overlap vanishes faster for higher physical dimensions as the rank of the tensor grows. This means the states become more "orthogonal" as we increase their dimension.

If we took the inner product of the same MPS we would be obtaining the square of the norm. In the case of the FiniteMPS.random() constructor, since it is canonically given as a normalized vector, we would recover 1.
