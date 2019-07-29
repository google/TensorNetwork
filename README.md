# TensorNetwork
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

- [TensorNetwork for Machine Learning](https://arxiv.org/abs/1906.06329)

The code for reproducing the results of these papers can be found in the [experiments](https://github.com/google/TensorNetwork/tree/master/experiments) directory.

## Installation
```
pip3 install tensornetwork
```

### To install on Docker

This will create a Docker image containing TensorNetwork. It will isolate a TensorNetwork installation from the rest of the system.

1. [Install Docker](https://docs.docker.com/install/#supported-platforms) on your host system.

2. Build the docker image for your system:
```bash
git clone https://github.com/google/TensorNetwork
cd TensorNetwork
docker build -t google/tensornetwork . # This builds the actual image based on latest Ubuntu, and installs TensorNetwork with the needed dependencies.
```

### To install on Docker for TensorNetwork development

To do your TensorNetwork development in a Docker virtual machine, you can use dev_tools/Dockerfile:

```bash
git clone https://github.com/google/TensorNetwork
cd TensorNetwork/dev_tools
docker build -t google/tensornetwork-dev . # This builds the actual image based on latest Ubuntu, cloning the TensorNetwork tree into it with the needed dependencies.
docker run -it google/tensornetwork-dev
```

If you want to contribute changes to TensorNetwork, you will instead want to fork the repository and submit pull requests from your fork.

## Documentation

For details about the TensorNetwork API, see the [reference documentation.](https://tensornetwork.readthedocs.io)

## Basic Example
Note: The following examples assume a TensorFlow v2 interface 
(in TF 1.13 or higher, run `tf.enable_v2_behavior()` after 
importing TensorFlow) but should also work with eager mode 
(`tf.enable_eager_execution()`). The actual library does work 
under graph mode, but documentation is limited.

Here, we build a simple 2 node contraction.
```python
import numpy as np
import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork

# Create the network
net = tensornetwork.TensorNetwork()
# Add the nodes
a = net.add_node(np.ones((10,), dtype=np.float32)) 
# Can use either np.array or tf.Tensor and can even mix them!
b = net.add_node(tf.ones((10,)))
edge = net.connect(a[0], b[0])
final_node = net.contract(edge)
print(final_node.tensor.numpy()) # Should print 10.0
```

## Optimized Contractions.
Usually, it is more computationally effective to flatten parallel edges before contracting them in order to avoid trace edges.
```python
net = tensornetwork.TensorNetwork()
a = net.add_node(tf.ones((2, 2, 2)))
b = net.add_node(tf.ones((2, 2, 2)))
e1 = net.connect(a[0], b[0])
# Edge contraction is communative, so the order doesn't matter.
e2 = net.connect(b[1], a[1])
e3 = net.connect(a[2], b[2])

flattened_edge = net.flatten_edges([e1, e2, e3])
print(net.contract(flattened_edge).tensor.numpy())
```
We also have `contract_between` and `contract_parallel` for your convenience. 

```python
# Contract all of the edges between a and b.
net.contract_between(a, b)
# Contract all of edges that are parallel to edge 
# (parallel means connected to the same nodes).
net.contract_parallel(edge)
```

## Split Node
You can split a node by doing a singular value decomposition. 
```python
# This will return two nodes and a tensor of the truncation error.
# The two nodes are the unitary matricies multiplied by the square root of the
# singular values.
# The `left_edges` are the edges that will end up on the `u_s` node, and `right_edges`
# will be on the `vh_s` node.
u_s, vh_s, trun_error = net.split_node(node, left_edges, right_edges)
# If you want the singular values in it's own node, you can use `split_node_full_svd`.
u, s, vh, trun_error = net.split_node_full_svd(node, left_edges, right_edges)
```

## Node and Edge names.
You can optionally name your nodes/edges. This can be useful for debugging, 
as all error messages will print the name of the broken edge/node.
```python
net = tensornetwork.TensorNetwork()
node = net.add_node(np.eye(2), name="Identity Matrix")
print("Name of node: {}".format(node.name))
edge = net.connect(node[0], node[1], name="Trace Edge")
print("Name of the edge: {}".format(edge.name))
# Adding name to a contraction will add the name to the new edge created.
final_result = net.contract(edge, name="Trace Of Identity")
print("Name of new node after contraction: {}".format(final_result.name))
```

## Named axes.
To make remembering what an axis does easier, you can optionally name a node's axes.
```python
net = tensornetwork.TensorNetwork()
a = net.add_node(np.zeros((2, 2)), axis_names=["alpha", "beta"])
edge = net.connect(a["beta"], a["alpha"])
```

## Edge reordering.
To assert that your result's axes are in the correct order, you can reorder a node at any time during computation.
```python
net = tensornetwork.TensorNetwork()
a = net.add_node(np.zeros((1, 2, 3)))
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
a = tf.random_normal((2,2))
b = tf.random_normal((2,2))
c = ncon([a,b], [(-1,1),(1,-2)])
print(tf.norm(tf.matmul(a,b) - c)) # Should be zero
```
It is also possible to generate a `TensorNetwork`:
```python
from tensornetwork import ncon_network
a = tf.random_normal((2,2))
b = tf.random_normal((2,2))
net, e_con, e_out = ncon_network([a,b], [(-1,1),(1,-2)])
for e in e_con:
    n = net.contract(e) # Contract edges in order
n.reorder_edges(e_out) # Permute final tensor as necessary
print(tf.norm(tf.matmul(a,b) - n.tensor))
```

## Different backend support.
Currently, we support TensorFlow, JAX, and NumPy as TensorNetwork backends. 

To change the default global backend, you can do:
```python
tensornetwork.set_default_backend("jax") # numpy, tensorflow, pytorch
```
Or, if you only want to change the backend for a single `TensorNetwork`, you can do:
```python
tensornetwork.TensorNetwork(backend="jax")
```
## Advanced examples
Some more sophisticated examples can be found under `examples/`.
### Trotter evolution of a wavefunction
Demonstrates time-evolution of a wavefunction, achieved by applying a quantum circuit
derived from a Trotter decomposition of the propagator. To run from source, use
```
python -m examples.wavefunctions.evolution_example
```
from the root directory.

## Disclaimer
This library is in *alpha* and will be going through a lot of breaking changes. While releases will be stable enough for research, we do not recommend using this in any production environment yet.

TensorNetwork is not an official Google product. Copyright 2019 The TensorNetwork Developers.
