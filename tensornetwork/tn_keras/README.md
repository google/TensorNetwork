# TensorNetwork Keras Layers

TN Keras exists to simplify tensorization of existing TensorFlow models. These layers try to match the APIs for existing Keras layers closely.

## Table of Contents

- [Usage](#usage)
- [Networks](#networks)
- [Support](#support)

## Usage

`pip install tensornetwork` and then:

```sh
import tensornetwork as tn
import tensorflow as tf
from tensornetwork.tn_keras import DenseMPO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a fully connected network using TN layer DenseMPO
mpo_model = Sequential()
mpo_model.add(DenseMPO(256, num_nodes=4, bond_dim=8, use_bias=True, activation='relu', input_shape=(1296,)))
mpo_model.add(DenseMPO(81, num_nodes=4, bond_dim=4, use_bias=True, activation='relu'))
mpo_model.add(Dense(1, use_bias=True, activation='sigmoid'))

...
```
## Networks

- **DenseDecomp**. A TN layer comparable to Dense that carries out matrix multiplication with 2 significantly smaller weight matrices instead of 1 large one. This layer is similar to performing a SVD on the weight matrix and dropping the lowest singular values. The TN looks like:
![Image of Decomp](https://4.bp.blogspot.com/-WCz7EdJ_1xU/XkHJD3UaefI/AAAAAAAACrg/rXcx1rF_2OomT04XI2topzLd2bBAcypXgCLcBGAsYHQ/s1600/imageLikeEmbed.png)

- **DenseMPO**. A TN layer that implements an MPO (Matrix Product Operator), a common tensor network found in condensed matter physics. MPOs are one of the most successful TNs we've seen in practice. Note for this layer the input dimension, output dimension, and number of nodes must all relate in order for the network structure to work. Specifically, `input_shape[-1]**(1. / num_nodes)` and `output_dim**(1. / num_nodes)` must both be round. The TN looks like:
![Image of MPO](https://1.bp.blogspot.com/-0-63SOqomZ0/XkL-3fcFdTI/AAAAAAAACsI/aK7hJf1PzRIGjV42qxA8cCUbjjj-9zRNwCLcBGAsYHQ/s1600/Screen%2BShot%2B2020-02-11%2Bat%2B11.21.38%2BAM.png)

## Support

Please [open an issue](https://github.com/google/TensorNetwork/issues/new) for support.