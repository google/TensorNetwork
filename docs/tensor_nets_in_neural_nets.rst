TensorNetworks in Neural Networks.
==================================

Here, we have a small toy example of how to use a TN inside of a fully
connected neural network.

First off, let’s install tensornetwork

.. code:: 

    !pip install tensornetwork
    
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    tf.enable_v2_behavior()
    # Import tensornetwork
    import tensornetwork as tn
    # Set the backend to tesorflow
    # (default is numpy)
    tn.set_default_backend("tensorflow")


.. parsed-literal::

    Requirement already satisfied: tensornetwork in /usr/local/lib/python3.6/dist-packages (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.6/dist-packages (from tensornetwork) (2.10.0)
    Requirement already satisfied: opt-einsum>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensornetwork) (3.1.0)
    Requirement already satisfied: numpy>=1.16 in /usr/local/lib/python3.6/dist-packages (from tensornetwork) (1.17.4)
    Requirement already satisfied: graphviz>=0.11.1 in /usr/local/lib/python3.6/dist-packages (from tensornetwork) (0.13.2)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from h5py>=2.9.0->tensornetwork) (1.12.0)



.. raw:: html

    <p style="color: red;">
    The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>
    We recommend you <a href="https://www.tensorflow.org/guide/migrate" target="_blank">upgrade</a> now 
    or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:
    <a href="https://colab.research.google.com/notebooks/tensorflow_version.ipynb" target="_blank">more info</a>.</p>



TensorNetwork layer definition
==============================

Here, we define the TensorNetwork layer we wish to use to replace the
fully connected layer. Here, we simply use a 2 node Matrix Product
Operator network to replace the normal dense weight matrix.

We TensorNetwork’s NCon API to keep the code short.

.. code:: 

    
    class TNLayer(tf.keras.layers.Layer):
    
      def __init__(self):
        super(TNLayer, self).__init__()
        # Create the variables for the layer.
        self.a_var = tf.Variable(tf.random.normal(
                shape=(8, 8, 2), stddev=1.0/16.0),
                 name="a", trainable=True)
        self.b_var = tf.Variable(tf.random.normal(shape=(8, 8, 2), stddev=1.0/16.0),
                                 name="b", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(8, 8)), name="bias", trainable=True)
    
      def call(self, inputs):
        # Define the contraction.
        # We break it out so we can parallelize a batch using
        # tf.vectorized_map (see below).
        def f(input_vec, a_var, b_var, bias_var):
          # Reshape to a matrix instead of a vector.
          input_vec = tf.reshape(input_vec, (8,8))
    
          # Now we create the network.
          a = tn.Node(a_var)
          b = tn.Node(b_var)
          x_node = tn.Node(input_vec)
          a[1] ^ x_node[0]
          b[1] ^ x_node[1]
          a[2] ^ b[2]
    
          # The TN should now look like this
          #   |     |
          #   a --- b
          #    \   /
          #      x
    
          # Now we begin the contraction.
          c = a @ x_node
          result = (c @ b).tensor
    
          # To make the code shorter, we also could've used Ncon.
          # The above few lines of code is the same as this:
          # result = tn.ncon([x, a_var, b_var], [[1, 2], [-1, 1, 3], [-2, 2, 3]])
    
          # Finally, add bias.
          return result + bias_var
      
        # To deal with a batch of items, we can use the tf.vectorized_map
        # function.
        # https://www.tensorflow.org/api_docs/python/tf/vectorized_map
        result = tf.vectorized_map(
            lambda vec: f(vec, self.a_var, self.b_var, self.bias), inputs)
        return tf.nn.swish(tf.reshape(result, (-1, 64)))

Smaller model
=============

These two models are effectively the same, but notice how the TN layer
has nearly 10x fewer parameters.

.. code:: 

    Dense = tf.keras.layers.Dense
    fc_model = tf.keras.Sequential(
        [
         tf.keras.Input(shape=(2,)),
         Dense(64, activation=tf.nn.swish),
         Dense(64, activation=tf.nn.swish),
         Dense(1, activation=None)])
    fc_model.summary()


.. parsed-literal::

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 64)                192       
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 4,417
    Trainable params: 4,417
    Non-trainable params: 0
    _________________________________________________________________


.. code:: 

    tn_model = tf.keras.Sequential(
        [
         tf.keras.Input(shape=(2,)),
         Dense(64, activation=tf.nn.swish),
         # Here, we replace the dense layer with our MPS.
         TNLayer(),
         Dense(1, activation=None)])
    tn_model.summary()


.. parsed-literal::

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 64)                192       
    _________________________________________________________________
    tn_layer (TNLayer)           (None, 64)                320       
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 577
    Trainable params: 577
    Non-trainable params: 0
    _________________________________________________________________


Training a model
================

You can train the TN model just as you would a normal neural network
model! Here, we give an example of how to do it in Keras.

.. code:: 

    X = np.concatenate([np.random.randn(20, 2) + np.array([3, 3]), 
                 np.random.randn(20, 2) + np.array([-3, -3]), 
                 np.random.randn(20, 2) + np.array([-3, 3]), 
                 np.random.randn(20, 2) + np.array([3, -3]),])
    
    Y = np.concatenate([np.ones((40)), -np.ones((40))])

.. code:: 

    tn_model.compile(optimizer="adam", loss="mean_squared_error")
    tn_model.fit(X, Y, epochs=300, verbose=1)


.. parsed-literal::

    Train on 80 samples
    Epoch 1/300


.. parsed-literal::

    /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "


.. parsed-literal::

    80/80 [==============================] - 1s 12ms/sample - loss: 1.0050


.. parsed-literal::

    /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "


.. parsed-literal::

    Epoch 2/300
    80/80 [==============================] - 0s 253us/sample - loss: 0.9992
    Epoch 3/300
    80/80 [==============================] - 0s 164us/sample - loss: 0.9926
    Epoch 4/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.9870
    Epoch 5/300
    80/80 [==============================] - 0s 181us/sample - loss: 0.9810
    Epoch 6/300
    80/80 [==============================] - 0s 150us/sample - loss: 0.9752
    Epoch 7/300
    80/80 [==============================] - 0s 173us/sample - loss: 0.9684
    Epoch 8/300
    80/80 [==============================] - 0s 191us/sample - loss: 0.9608
    Epoch 9/300
    80/80 [==============================] - 0s 157us/sample - loss: 0.9527
    Epoch 10/300
    80/80 [==============================] - 0s 235us/sample - loss: 0.9433
    Epoch 11/300
    80/80 [==============================] - 0s 188us/sample - loss: 0.9331
    Epoch 12/300
    80/80 [==============================] - 0s 166us/sample - loss: 0.9209
    Epoch 13/300
    80/80 [==============================] - 0s 219us/sample - loss: 0.9075
    Epoch 14/300
    80/80 [==============================] - 0s 226us/sample - loss: 0.8920
    Epoch 15/300
    80/80 [==============================] - 0s 202us/sample - loss: 0.8744
    Epoch 16/300
    80/80 [==============================] - 0s 190us/sample - loss: 0.8545
    Epoch 17/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.8319
    Epoch 18/300
    80/80 [==============================] - 0s 187us/sample - loss: 0.8063
    Epoch 19/300
    80/80 [==============================] - 0s 184us/sample - loss: 0.7774
    Epoch 20/300
    80/80 [==============================] - 0s 253us/sample - loss: 0.7451
    Epoch 21/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.7095
    Epoch 22/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.6699
    Epoch 23/300
    80/80 [==============================] - 0s 188us/sample - loss: 0.6265
    Epoch 24/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.5792
    Epoch 25/300
    80/80 [==============================] - 0s 181us/sample - loss: 0.5289
    Epoch 26/300
    80/80 [==============================] - 0s 167us/sample - loss: 0.4754
    Epoch 27/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.4211
    Epoch 28/300
    80/80 [==============================] - 0s 222us/sample - loss: 0.3659
    Epoch 29/300
    80/80 [==============================] - 0s 205us/sample - loss: 0.3112
    Epoch 30/300
    80/80 [==============================] - 0s 167us/sample - loss: 0.2601
    Epoch 31/300
    80/80 [==============================] - 0s 252us/sample - loss: 0.2149
    Epoch 32/300
    80/80 [==============================] - 0s 191us/sample - loss: 0.1767
    Epoch 33/300
    80/80 [==============================] - 0s 177us/sample - loss: 0.1495
    Epoch 34/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.1301
    Epoch 35/300
    80/80 [==============================] - 0s 254us/sample - loss: 0.1211
    Epoch 36/300
    80/80 [==============================] - 0s 336us/sample - loss: 0.1191
    Epoch 37/300
    80/80 [==============================] - 0s 261us/sample - loss: 0.1219
    Epoch 38/300
    80/80 [==============================] - 0s 209us/sample - loss: 0.1237
    Epoch 39/300
    80/80 [==============================] - 0s 204us/sample - loss: 0.1248
    Epoch 40/300
    80/80 [==============================] - 0s 225us/sample - loss: 0.1235
    Epoch 41/300
    80/80 [==============================] - 0s 239us/sample - loss: 0.1217
    Epoch 42/300
    80/80 [==============================] - 0s 166us/sample - loss: 0.1199
    Epoch 43/300
    80/80 [==============================] - 0s 155us/sample - loss: 0.1184
    Epoch 44/300
    80/80 [==============================] - 0s 190us/sample - loss: 0.1167
    Epoch 45/300
    80/80 [==============================] - 0s 206us/sample - loss: 0.1169
    Epoch 46/300
    80/80 [==============================] - 0s 188us/sample - loss: 0.1159
    Epoch 47/300
    80/80 [==============================] - 0s 170us/sample - loss: 0.1154
    Epoch 48/300
    80/80 [==============================] - 0s 159us/sample - loss: 0.1154
    Epoch 49/300
    80/80 [==============================] - 0s 180us/sample - loss: 0.1152
    Epoch 50/300
    80/80 [==============================] - 0s 218us/sample - loss: 0.1148
    Epoch 51/300
    80/80 [==============================] - 0s 200us/sample - loss: 0.1145
    Epoch 52/300
    80/80 [==============================] - 0s 395us/sample - loss: 0.1143
    Epoch 53/300
    80/80 [==============================] - 0s 238us/sample - loss: 0.1142
    Epoch 54/300
    80/80 [==============================] - 0s 248us/sample - loss: 0.1143
    Epoch 55/300
    80/80 [==============================] - 0s 287us/sample - loss: 0.1138
    Epoch 56/300
    80/80 [==============================] - 0s 178us/sample - loss: 0.1133
    Epoch 57/300
    80/80 [==============================] - 0s 236us/sample - loss: 0.1127
    Epoch 58/300
    80/80 [==============================] - 0s 254us/sample - loss: 0.1126
    Epoch 59/300
    80/80 [==============================] - 0s 264us/sample - loss: 0.1128
    Epoch 60/300
    80/80 [==============================] - 0s 188us/sample - loss: 0.1122
    Epoch 61/300
    80/80 [==============================] - 0s 278us/sample - loss: 0.1121
    Epoch 62/300
    80/80 [==============================] - 0s 210us/sample - loss: 0.1121
    Epoch 63/300
    80/80 [==============================] - 0s 224us/sample - loss: 0.1114
    Epoch 64/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.1109
    Epoch 65/300
    80/80 [==============================] - 0s 219us/sample - loss: 0.1115
    Epoch 66/300
    80/80 [==============================] - 0s 243us/sample - loss: 0.1108
    Epoch 67/300
    80/80 [==============================] - 0s 287us/sample - loss: 0.1108
    Epoch 68/300
    80/80 [==============================] - 0s 221us/sample - loss: 0.1103
    Epoch 69/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.1102
    Epoch 70/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.1100
    Epoch 71/300
    80/80 [==============================] - 0s 236us/sample - loss: 0.1093
    Epoch 72/300
    80/80 [==============================] - 0s 215us/sample - loss: 0.1090
    Epoch 73/300
    80/80 [==============================] - 0s 225us/sample - loss: 0.1088
    Epoch 74/300
    80/80 [==============================] - 0s 250us/sample - loss: 0.1086
    Epoch 75/300
    80/80 [==============================] - 0s 213us/sample - loss: 0.1084
    Epoch 76/300
    80/80 [==============================] - 0s 214us/sample - loss: 0.1081
    Epoch 77/300
    80/80 [==============================] - 0s 216us/sample - loss: 0.1078
    Epoch 78/300
    80/80 [==============================] - 0s 207us/sample - loss: 0.1077
    Epoch 79/300
    80/80 [==============================] - 0s 166us/sample - loss: 0.1076
    Epoch 80/300
    80/80 [==============================] - 0s 212us/sample - loss: 0.1071
    Epoch 81/300
    80/80 [==============================] - 0s 218us/sample - loss: 0.1072
    Epoch 82/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.1071
    Epoch 83/300
    80/80 [==============================] - 0s 210us/sample - loss: 0.1070
    Epoch 84/300
    80/80 [==============================] - 0s 184us/sample - loss: 0.1067
    Epoch 85/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.1064
    Epoch 86/300
    80/80 [==============================] - 0s 216us/sample - loss: 0.1061
    Epoch 87/300
    80/80 [==============================] - 0s 192us/sample - loss: 0.1056
    Epoch 88/300
    80/80 [==============================] - 0s 217us/sample - loss: 0.1051
    Epoch 89/300
    80/80 [==============================] - 0s 185us/sample - loss: 0.1047
    Epoch 90/300
    80/80 [==============================] - 0s 220us/sample - loss: 0.1044
    Epoch 91/300
    80/80 [==============================] - 0s 250us/sample - loss: 0.1043
    Epoch 92/300
    80/80 [==============================] - 0s 330us/sample - loss: 0.1039
    Epoch 93/300
    80/80 [==============================] - 0s 222us/sample - loss: 0.1037
    Epoch 94/300
    80/80 [==============================] - 0s 238us/sample - loss: 0.1034
    Epoch 95/300
    80/80 [==============================] - 0s 215us/sample - loss: 0.1032
    Epoch 96/300
    80/80 [==============================] - 0s 244us/sample - loss: 0.1029
    Epoch 97/300
    80/80 [==============================] - 0s 151us/sample - loss: 0.1027
    Epoch 98/300
    80/80 [==============================] - 0s 186us/sample - loss: 0.1024
    Epoch 99/300
    80/80 [==============================] - 0s 283us/sample - loss: 0.1019
    Epoch 100/300
    80/80 [==============================] - 0s 277us/sample - loss: 0.1016
    Epoch 101/300
    80/80 [==============================] - 0s 235us/sample - loss: 0.1012
    Epoch 102/300
    80/80 [==============================] - 0s 292us/sample - loss: 0.1010
    Epoch 103/300
    80/80 [==============================] - 0s 178us/sample - loss: 0.1007
    Epoch 104/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.1005
    Epoch 105/300
    80/80 [==============================] - 0s 203us/sample - loss: 0.1004
    Epoch 106/300
    80/80 [==============================] - 0s 174us/sample - loss: 0.1000
    Epoch 107/300
    80/80 [==============================] - 0s 225us/sample - loss: 0.0995
    Epoch 108/300
    80/80 [==============================] - 0s 223us/sample - loss: 0.0988
    Epoch 109/300
    80/80 [==============================] - 0s 195us/sample - loss: 0.0988
    Epoch 110/300
    80/80 [==============================] - 0s 208us/sample - loss: 0.0984
    Epoch 111/300
    80/80 [==============================] - 0s 172us/sample - loss: 0.0983
    Epoch 112/300
    80/80 [==============================] - 0s 207us/sample - loss: 0.0978
    Epoch 113/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.0974
    Epoch 114/300
    80/80 [==============================] - 0s 242us/sample - loss: 0.0972
    Epoch 115/300
    80/80 [==============================] - 0s 271us/sample - loss: 0.0968
    Epoch 116/300
    80/80 [==============================] - 0s 199us/sample - loss: 0.0962
    Epoch 117/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.0956
    Epoch 118/300
    80/80 [==============================] - 0s 229us/sample - loss: 0.0954
    Epoch 119/300
    80/80 [==============================] - 0s 198us/sample - loss: 0.0953
    Epoch 120/300
    80/80 [==============================] - 0s 218us/sample - loss: 0.0948
    Epoch 121/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.0943
    Epoch 122/300
    80/80 [==============================] - 0s 247us/sample - loss: 0.0941
    Epoch 123/300
    80/80 [==============================] - 0s 265us/sample - loss: 0.0937
    Epoch 124/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.0930
    Epoch 125/300
    80/80 [==============================] - 0s 185us/sample - loss: 0.0927
    Epoch 126/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.0923
    Epoch 127/300
    80/80 [==============================] - 0s 187us/sample - loss: 0.0917
    Epoch 128/300
    80/80 [==============================] - 0s 197us/sample - loss: 0.0914
    Epoch 129/300
    80/80 [==============================] - 0s 185us/sample - loss: 0.0911
    Epoch 130/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.0904
    Epoch 131/300
    80/80 [==============================] - 0s 204us/sample - loss: 0.0898
    Epoch 132/300
    80/80 [==============================] - 0s 186us/sample - loss: 0.0897
    Epoch 133/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.0891
    Epoch 134/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.0886
    Epoch 135/300
    80/80 [==============================] - 0s 224us/sample - loss: 0.0882
    Epoch 136/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.0876
    Epoch 137/300
    80/80 [==============================] - 0s 200us/sample - loss: 0.0872
    Epoch 138/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.0866
    Epoch 139/300
    80/80 [==============================] - 0s 215us/sample - loss: 0.0858
    Epoch 140/300
    80/80 [==============================] - 0s 299us/sample - loss: 0.0853
    Epoch 141/300
    80/80 [==============================] - 0s 192us/sample - loss: 0.0848
    Epoch 142/300
    80/80 [==============================] - 0s 221us/sample - loss: 0.0846
    Epoch 143/300
    80/80 [==============================] - 0s 184us/sample - loss: 0.0837
    Epoch 144/300
    80/80 [==============================] - 0s 224us/sample - loss: 0.0833
    Epoch 145/300
    80/80 [==============================] - 0s 236us/sample - loss: 0.0826
    Epoch 146/300
    80/80 [==============================] - 0s 300us/sample - loss: 0.0820
    Epoch 147/300
    80/80 [==============================] - 0s 186us/sample - loss: 0.0821
    Epoch 148/300
    80/80 [==============================] - 0s 150us/sample - loss: 0.0813
    Epoch 149/300
    80/80 [==============================] - 0s 214us/sample - loss: 0.0807
    Epoch 150/300
    80/80 [==============================] - 0s 158us/sample - loss: 0.0796
    Epoch 151/300
    80/80 [==============================] - 0s 268us/sample - loss: 0.0791
    Epoch 152/300
    80/80 [==============================] - 0s 198us/sample - loss: 0.0786
    Epoch 153/300
    80/80 [==============================] - 0s 184us/sample - loss: 0.0781
    Epoch 154/300
    80/80 [==============================] - 0s 158us/sample - loss: 0.0772
    Epoch 155/300
    80/80 [==============================] - 0s 193us/sample - loss: 0.0768
    Epoch 156/300
    80/80 [==============================] - 0s 226us/sample - loss: 0.0762
    Epoch 157/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.0760
    Epoch 158/300
    80/80 [==============================] - 0s 223us/sample - loss: 0.0753
    Epoch 159/300
    80/80 [==============================] - 0s 245us/sample - loss: 0.0743
    Epoch 160/300
    80/80 [==============================] - 0s 234us/sample - loss: 0.0738
    Epoch 161/300
    80/80 [==============================] - 0s 192us/sample - loss: 0.0731
    Epoch 162/300
    80/80 [==============================] - 0s 188us/sample - loss: 0.0727
    Epoch 163/300
    80/80 [==============================] - 0s 219us/sample - loss: 0.0721
    Epoch 164/300
    80/80 [==============================] - 0s 190us/sample - loss: 0.0711
    Epoch 165/300
    80/80 [==============================] - 0s 251us/sample - loss: 0.0702
    Epoch 166/300
    80/80 [==============================] - 0s 286us/sample - loss: 0.0703
    Epoch 167/300
    80/80 [==============================] - 0s 227us/sample - loss: 0.0696
    Epoch 168/300
    80/80 [==============================] - 0s 171us/sample - loss: 0.0688
    Epoch 169/300
    80/80 [==============================] - 0s 172us/sample - loss: 0.0677
    Epoch 170/300
    80/80 [==============================] - 0s 186us/sample - loss: 0.0674
    Epoch 171/300
    80/80 [==============================] - 0s 195us/sample - loss: 0.0668
    Epoch 172/300
    80/80 [==============================] - 0s 221us/sample - loss: 0.0660
    Epoch 173/300
    80/80 [==============================] - 0s 232us/sample - loss: 0.0652
    Epoch 174/300
    80/80 [==============================] - 0s 229us/sample - loss: 0.0650
    Epoch 175/300
    80/80 [==============================] - 0s 229us/sample - loss: 0.0639
    Epoch 176/300
    80/80 [==============================] - 0s 241us/sample - loss: 0.0632
    Epoch 177/300
    80/80 [==============================] - 0s 254us/sample - loss: 0.0626
    Epoch 178/300
    80/80 [==============================] - 0s 198us/sample - loss: 0.0619
    Epoch 179/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.0611
    Epoch 180/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.0604
    Epoch 181/300
    80/80 [==============================] - 0s 180us/sample - loss: 0.0598
    Epoch 182/300
    80/80 [==============================] - 0s 335us/sample - loss: 0.0592
    Epoch 183/300
    80/80 [==============================] - 0s 213us/sample - loss: 0.0587
    Epoch 184/300
    80/80 [==============================] - 0s 187us/sample - loss: 0.0579
    Epoch 185/300
    80/80 [==============================] - 0s 200us/sample - loss: 0.0571
    Epoch 186/300
    80/80 [==============================] - 0s 223us/sample - loss: 0.0565
    Epoch 187/300
    80/80 [==============================] - 0s 214us/sample - loss: 0.0559
    Epoch 188/300
    80/80 [==============================] - 0s 223us/sample - loss: 0.0553
    Epoch 189/300
    80/80 [==============================] - 0s 213us/sample - loss: 0.0547
    Epoch 190/300
    80/80 [==============================] - 0s 204us/sample - loss: 0.0539
    Epoch 191/300
    80/80 [==============================] - 0s 231us/sample - loss: 0.0533
    Epoch 192/300
    80/80 [==============================] - 0s 217us/sample - loss: 0.0527
    Epoch 193/300
    80/80 [==============================] - 0s 190us/sample - loss: 0.0520
    Epoch 194/300
    80/80 [==============================] - 0s 222us/sample - loss: 0.0514
    Epoch 195/300
    80/80 [==============================] - 0s 219us/sample - loss: 0.0509
    Epoch 196/300
    80/80 [==============================] - 0s 177us/sample - loss: 0.0502
    Epoch 197/300
    80/80 [==============================] - 0s 249us/sample - loss: 0.0495
    Epoch 198/300
    80/80 [==============================] - 0s 188us/sample - loss: 0.0489
    Epoch 199/300
    80/80 [==============================] - 0s 177us/sample - loss: 0.0485
    Epoch 200/300
    80/80 [==============================] - 0s 180us/sample - loss: 0.0478
    Epoch 201/300
    80/80 [==============================] - 0s 172us/sample - loss: 0.0472
    Epoch 202/300
    80/80 [==============================] - 0s 185us/sample - loss: 0.0465
    Epoch 203/300
    80/80 [==============================] - 0s 291us/sample - loss: 0.0462
    Epoch 204/300
    80/80 [==============================] - 0s 233us/sample - loss: 0.0456
    Epoch 205/300
    80/80 [==============================] - 0s 210us/sample - loss: 0.0453
    Epoch 206/300
    80/80 [==============================] - 0s 214us/sample - loss: 0.0445
    Epoch 207/300
    80/80 [==============================] - 0s 217us/sample - loss: 0.0439
    Epoch 208/300
    80/80 [==============================] - 0s 168us/sample - loss: 0.0436
    Epoch 209/300
    80/80 [==============================] - 0s 161us/sample - loss: 0.0431
    Epoch 210/300
    80/80 [==============================] - 0s 229us/sample - loss: 0.0426
    Epoch 211/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.0420
    Epoch 212/300
    80/80 [==============================] - 0s 148us/sample - loss: 0.0416
    Epoch 213/300
    80/80 [==============================] - 0s 192us/sample - loss: 0.0413
    Epoch 214/300
    80/80 [==============================] - 0s 243us/sample - loss: 0.0409
    Epoch 215/300
    80/80 [==============================] - 0s 280us/sample - loss: 0.0401
    Epoch 216/300
    80/80 [==============================] - 0s 209us/sample - loss: 0.0398
    Epoch 217/300
    80/80 [==============================] - 0s 262us/sample - loss: 0.0395
    Epoch 218/300
    80/80 [==============================] - 0s 238us/sample - loss: 0.0392
    Epoch 219/300
    80/80 [==============================] - 0s 231us/sample - loss: 0.0387
    Epoch 220/300
    80/80 [==============================] - 0s 229us/sample - loss: 0.0384
    Epoch 221/300
    80/80 [==============================] - 0s 171us/sample - loss: 0.0379
    Epoch 222/300
    80/80 [==============================] - 0s 175us/sample - loss: 0.0376
    Epoch 223/300
    80/80 [==============================] - 0s 194us/sample - loss: 0.0374
    Epoch 224/300
    80/80 [==============================] - 0s 201us/sample - loss: 0.0369
    Epoch 225/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.0367
    Epoch 226/300
    80/80 [==============================] - 0s 197us/sample - loss: 0.0365
    Epoch 227/300
    80/80 [==============================] - 0s 193us/sample - loss: 0.0362
    Epoch 228/300
    80/80 [==============================] - 0s 170us/sample - loss: 0.0359
    Epoch 229/300
    80/80 [==============================] - 0s 221us/sample - loss: 0.0355
    Epoch 230/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.0353
    Epoch 231/300
    80/80 [==============================] - 0s 195us/sample - loss: 0.0352
    Epoch 232/300
    80/80 [==============================] - 0s 194us/sample - loss: 0.0349
    Epoch 233/300
    80/80 [==============================] - 0s 209us/sample - loss: 0.0347
    Epoch 234/300
    80/80 [==============================] - 0s 215us/sample - loss: 0.0346
    Epoch 235/300
    80/80 [==============================] - 0s 257us/sample - loss: 0.0343
    Epoch 236/300
    80/80 [==============================] - 0s 171us/sample - loss: 0.0341
    Epoch 237/300
    80/80 [==============================] - 0s 197us/sample - loss: 0.0340
    Epoch 238/300
    80/80 [==============================] - 0s 190us/sample - loss: 0.0339
    Epoch 239/300
    80/80 [==============================] - 0s 181us/sample - loss: 0.0337
    Epoch 240/300
    80/80 [==============================] - 0s 205us/sample - loss: 0.0333
    Epoch 241/300
    80/80 [==============================] - 0s 163us/sample - loss: 0.0331
    Epoch 242/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.0330
    Epoch 243/300
    80/80 [==============================] - 0s 159us/sample - loss: 0.0327
    Epoch 244/300
    80/80 [==============================] - 0s 173us/sample - loss: 0.0326
    Epoch 245/300
    80/80 [==============================] - 0s 196us/sample - loss: 0.0326
    Epoch 246/300
    80/80 [==============================] - 0s 204us/sample - loss: 0.0324
    Epoch 247/300
    80/80 [==============================] - 0s 171us/sample - loss: 0.0323
    Epoch 248/300
    80/80 [==============================] - 0s 246us/sample - loss: 0.0321
    Epoch 249/300
    80/80 [==============================] - 0s 169us/sample - loss: 0.0319
    Epoch 250/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.0319
    Epoch 251/300
    80/80 [==============================] - 0s 193us/sample - loss: 0.0317
    Epoch 252/300
    80/80 [==============================] - 0s 240us/sample - loss: 0.0315
    Epoch 253/300
    80/80 [==============================] - 0s 211us/sample - loss: 0.0315
    Epoch 254/300
    80/80 [==============================] - 0s 168us/sample - loss: 0.0314
    Epoch 255/300
    80/80 [==============================] - 0s 212us/sample - loss: 0.0313
    Epoch 256/300
    80/80 [==============================] - 0s 234us/sample - loss: 0.0312
    Epoch 257/300
    80/80 [==============================] - 0s 216us/sample - loss: 0.0313
    Epoch 258/300
    80/80 [==============================] - 0s 195us/sample - loss: 0.0311
    Epoch 259/300
    80/80 [==============================] - 0s 224us/sample - loss: 0.0308
    Epoch 260/300
    80/80 [==============================] - 0s 227us/sample - loss: 0.0308
    Epoch 261/300
    80/80 [==============================] - 0s 202us/sample - loss: 0.0306
    Epoch 262/300
    80/80 [==============================] - 0s 259us/sample - loss: 0.0306
    Epoch 263/300
    80/80 [==============================] - 0s 263us/sample - loss: 0.0305
    Epoch 264/300
    80/80 [==============================] - 0s 183us/sample - loss: 0.0303
    Epoch 265/300
    80/80 [==============================] - 0s 193us/sample - loss: 0.0303
    Epoch 266/300
    80/80 [==============================] - 0s 185us/sample - loss: 0.0303
    Epoch 267/300
    80/80 [==============================] - 0s 226us/sample - loss: 0.0303
    Epoch 268/300
    80/80 [==============================] - 0s 222us/sample - loss: 0.0302
    Epoch 269/300
    80/80 [==============================] - 0s 222us/sample - loss: 0.0303
    Epoch 270/300
    80/80 [==============================] - 0s 192us/sample - loss: 0.0302
    Epoch 271/300
    80/80 [==============================] - 0s 206us/sample - loss: 0.0300
    Epoch 272/300
    80/80 [==============================] - 0s 208us/sample - loss: 0.0300
    Epoch 273/300
    80/80 [==============================] - 0s 196us/sample - loss: 0.0299
    Epoch 274/300
    80/80 [==============================] - 0s 192us/sample - loss: 0.0298
    Epoch 275/300
    80/80 [==============================] - 0s 181us/sample - loss: 0.0298
    Epoch 276/300
    80/80 [==============================] - 0s 187us/sample - loss: 0.0298
    Epoch 277/300
    80/80 [==============================] - 0s 165us/sample - loss: 0.0298
    Epoch 278/300
    80/80 [==============================] - 0s 223us/sample - loss: 0.0296
    Epoch 279/300
    80/80 [==============================] - 0s 187us/sample - loss: 0.0296
    Epoch 280/300
    80/80 [==============================] - 0s 261us/sample - loss: 0.0297
    Epoch 281/300
    80/80 [==============================] - 0s 236us/sample - loss: 0.0295
    Epoch 282/300
    80/80 [==============================] - 0s 198us/sample - loss: 0.0295
    Epoch 283/300
    80/80 [==============================] - 0s 175us/sample - loss: 0.0293
    Epoch 284/300
    80/80 [==============================] - 0s 224us/sample - loss: 0.0293
    Epoch 285/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.0292
    Epoch 286/300
    80/80 [==============================] - 0s 191us/sample - loss: 0.0295
    Epoch 287/300
    80/80 [==============================] - 0s 213us/sample - loss: 0.0292
    Epoch 288/300
    80/80 [==============================] - 0s 168us/sample - loss: 0.0293
    Epoch 289/300
    80/80 [==============================] - 0s 219us/sample - loss: 0.0290
    Epoch 290/300
    80/80 [==============================] - 0s 246us/sample - loss: 0.0290
    Epoch 291/300
    80/80 [==============================] - 0s 212us/sample - loss: 0.0291
    Epoch 292/300
    80/80 [==============================] - 0s 176us/sample - loss: 0.0292
    Epoch 293/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.0289
    Epoch 294/300
    80/80 [==============================] - 0s 150us/sample - loss: 0.0290
    Epoch 295/300
    80/80 [==============================] - 0s 258us/sample - loss: 0.0289
    Epoch 296/300
    80/80 [==============================] - 0s 207us/sample - loss: 0.0289
    Epoch 297/300
    80/80 [==============================] - 0s 189us/sample - loss: 0.0288
    Epoch 298/300
    80/80 [==============================] - 0s 216us/sample - loss: 0.0288
    Epoch 299/300
    80/80 [==============================] - 0s 296us/sample - loss: 0.0288
    Epoch 300/300
    80/80 [==============================] - 0s 207us/sample - loss: 0.0286




.. parsed-literal::

    <tensorflow.python.keras.callbacks.History at 0x7f5b361de438>



.. code:: 

    # Plotting code, feel free to ignore.
    h = 1.0
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # here "model" is your model's prediction (classification) function
    Z = tn_model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis('off')
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x7f5b365f38d0>




.. image:: Tensor_Networks_in_Neural_Networks_files/Tensor_Networks_in_Neural_Networks_10_1.png


VS Fully Connected
==================

Notice how the TN above has a much smoother decision boundary vs the
dense network. While the final training loss is slightly worse, the end
result is slightly better! Therefore, TN can be used as a form of
regularization too!

.. code:: 

    fc_model.compile(optimizer="adam", loss="mean_squared_error")
    fc_model.fit(X, Y, epochs=300, verbose=0)
    # Plotting code, feel free to ignore.
    h = 1.0
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # here "model" is your model's prediction (classification) function
    Z = fc_model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis('off')
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)





.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x7f5b365dcc88>




.. image:: Tensor_Networks_in_Neural_Networks_files/Tensor_Networks_in_Neural_Networks_12_1.png


