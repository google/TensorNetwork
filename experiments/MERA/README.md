# TensorNetwork
[![Build Status](https://travis-ci.org/google/TensorNetwork.svg?branch=master)](https://travis-ci.org/google/TensorNetwork)

#Multi-scale Entanglement Renormalization using TensorFlow
This is an implementation of a scale-invariant binary MERA and modified binary MERA optimization
for the transverse field Ising model at criticality.

To optimize a binary MERA and calculate scaling dimensions, run
```python
python -m experiments.MERA.binary_mera_example
```

To run benchmark for binary MERA and modified binary MERA, run 
```python
python -m experiments.MERA.modified_binary_mera
python -m experiments.MERA.binary_mera
```

## Credits
Special thanks to Glen Evenbly. Parts of the code are based on
his modified binary MERA implementation on www.tensors.net

