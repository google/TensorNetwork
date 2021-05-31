<img src="https://user-images.githubusercontent.com/8702042/67589472-5a1d0e80-f70d-11e9-8812-64647814ae96.png" width="60%" height="60%">

[![Build Status](https://travis-ci.org/google/TensorNetwork.svg?branch=master)](https://travis-ci.org/google/TensorNetwork)


A tensor network wrapper for TensorFlow, JAX, PyTorch, and Numpy.

For an overview of tensor networks please see the following: 

- [Matrices as Tensor Network Diagrams](https://www.math3ma.com/blog/matrices-as-tensor-network-diagrams)


- [Crash Course in Tensor Networks (video)](https://www.youtube.com/watch?v=YN2YBB0viKo)

- [Hand-waving and interpretive dance: an introductory course on tensor networks](https://iopscience.iop.org/article/10.1088/1751-8121/aa6dc3)

- [Tensor Networks in a Nutshell](https://arxiv.org/abs/1708.00006)

- [A Practical Introduction to Tensor Networks](https://arxiv.org/abs/1306.2164)

More information can be found in our TensorNetwork papers:

- [TensorNetwork: A Library for Physics and Machine Learning](https://arxiv.org/abs/1905.01330)

- [TensorNetwork on TensorFlow: A Spin Chain Application Using Tree Tensor Networks](https://arxiv.org/abs/1905.01331)

- [TensorNetwork on TensorFlow: Entanglement Renormalization for quantum critical lattice models](https://arxiv.org/abs/1906.12030)

- [TensorNetwork for Machine Learning](https://arxiv.org/abs/1906.06329)


## Installation
```
pip3 install tensornetwork
```

## Documentation

For details about the TensorNetwork API, see the [reference documentation.](https://tensornetwork.readthedocs.io)


## Tutorials

[Basic API tutorial](https://colab.research.google.com/drive/1Fp9DolkPT-P_Dkg_s9PLbTOKSq64EVSu)

[Tensor Networks inside Neural Networks using Keras](https://colab.research.google.com/github/google/TensorNetwork/blob/master/colabs/Tensor_Networks_in_Neural_Networks.ipynb)
## Basic Example

Here, we build a simple 2 node contraction.
```python
import numpy as np
import tensornetwork as tn

# Create the nodes
a = tn.Node(np.ones((10,))) 
b = tn.Node(np.ones((10,)))
edge = a[0] ^ b[0] # Equal to tn.connect(a[0], b[0])
final_node = tn.contract(edge)
print(final_node.tensor) # Should print 10.0
```

## Optimized Contractions.
Usually, it is more computationally effective to flatten parallel edges before contracting them in order to avoid trace edges.
We have `contract_between` and `contract_parallel` that do this automatically for your convenience. 

```python
# Contract all of the edges between a and b
# and create a new node `c`.
c = tn.contract_between(a, b)
# This is the same as above, but much shorter.
c = a @ b

# Contract all of edges that are parallel to edge 
# (parallel means connected to the same nodes).
c = tn.contract_parallel(edge)
```

## Split Node
You can split a node by doing a singular value decomposition. 
```python
# This will return two nodes and a tensor of the truncation error.
# The two nodes are the unitary matrices multiplied by the square root of the
# singular values.
# The `left_edges` are the edges that will end up on the `u_s` node, and `right_edges`
# will be on the `vh_s` node.
u_s, vh_s, trun_error = tn.split_node(node, left_edges, right_edges)
# If you want the singular values in it's own node, you can use `split_node_full_svd`.
u, s, vh, trun_error = tn.split_node_full_svd(node, left_edges, right_edges)
```

## Node and Edge names.
You can optionally name your nodes/edges. This can be useful for debugging, 
as all error messages will print the name of the broken edge/node.
```python
node = tn.Node(np.eye(2), name="Identity Matrix")
print("Name of node: {}".format(node.name))
edge = tn.connect(node[0], node[1], name="Trace Edge")
print("Name of the edge: {}".format(edge.name))
# Adding name to a contraction will add the name to the new edge created.
final_result = tn.contract(edge, name="Trace Of Identity")
print("Name of new node after contraction: {}".format(final_result.name))
```

## Named axes.
To make remembering what an axis does easier, you can optionally name a node's axes.
```python
a = tn.Node(np.zeros((2, 2)), axis_names=["alpha", "beta"])
edge = a["beta"] ^ a["alpha"]
```

## Edge reordering.
To assert that your result's axes are in the correct order, you can reorder a node at any time during computation.
```python
a = tn.Node(np.zeros((1, 2, 3)))
e1 = a[0]
e2 = a[1]
e3 = a[2]
a.reorder_edges([e3, e1, e2])
# If you already know the axis values, you can equivalently do
# a.reorder_axes([2, 0, 1])
print(a.tensor.shape) # Should print (3, 1, 2)
```

## NCON interface.
For a more compact specification of a tensor network and its contraction, there is `ncon()`. For example:
```python
from tensornetwork import ncon
a = np.ones((2, 2))
b = np.ones((2, 2))
c = ncon([a, b], [(-1, 1), (1, -2)])
print(c)
```

## Different backend support.
Currently, we support JAX, TensorFlow, PyTorch and NumPy as TensorNetwork backends.
We also support tensors with Abelian symmetries via a `symmetric` backend, see the [reference
documentation](https://tensornetwork.readthedocs.io/en/latest/block_sparse_tutorial.html) for more details.

To change the default global backend, you can do:
```python
tn.set_default_backend("jax") # tensorflow, pytorch, numpy, symmetric
```
Or, if you only want to change the backend for a single `Node`, you can do:
```python
tn.Node(tensor, backend="jax")
```

If you want to run your contractions on a GPU, we highly recommend using JAX, as it has the closet API to NumPy.

## Disclaimer
This library is in *alpha* and will be going through a lot of breaking changes. While releases will be stable enough for research, we do not recommend using this in any production environment yet.

TensorNetwork is not an official Google product. Copyright 2019 The TensorNetwork Developers.

## Citation
If you are using TensorNetwork for your research please cite this work using the following bibtex entry:

```
@misc{roberts2019tensornetwork,
      title={TensorNetwork: A Library for Physics and Machine Learning}, 
      author={Chase Roberts and Ashley Milsted and Martin Ganahl and Adam Zalcman and Bruce Fontaine and Yijian Zou and Jack Hidary and Guifre Vidal and Stefan Leichenauer},
      year={2019},
      eprint={1905.01330},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}
```

