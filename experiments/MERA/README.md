# Multi-scale Entanglement Renormalization using TensorFlow
This is an implementation of a scale-invariant binary MERA and modified binary MERA optimization
for the transverse field Ising model at criticality.

To optimize a binary MERA and calculate scaling dimensions, run
```python
python -m experiments.MERA.binary_mera_example
```
from the root directory.

To run benchmarks for binary MERA and modified binary MERA, run
```python
python -m experiments.MERA.modified_binary_mera
python -m experiments.MERA.binary_mera
```
from the root directory.

## Credits
Special thanks to Glen Evenbly. Parts of the code are based on
his modified binary MERA implementation on www.tensors.net

