TN Keras Layers
------------------

TN Keras exists to simplify tensorization of existing TensorFlow models. 
These layers try to match the APIs for existing Keras layers closely. 
Please note these layers are currently intended for experimentation only, 
not production. These layers are in alpha and upcoming releases might include 
breaking changes.

APIs are listed here. An overview of these layers is available 
`here <https://github.com/google/TensorNetwork/tree/master/tensornetwork/tn_keras>`_.

.. autosummary::
     :toctree: stubs

     tensornetwork.tn_keras.layers.DenseDecomp
	 tensornetwork.tn_keras.layers.DenseMPO
	 tensornetwork.tn_keras.layers.Conv2DMPO
	 tensornetwork.tn_keras.layers.DenseCondenser
	 tensornetwork.tn_keras.layers.DenseExpander
	 tensornetwork.tn_keras.layers.DenseEntangler
