import numpy as np 
import tensornetwork

net = tensornetwork.TensorNetwork("numpy")
a = net.add_node(np.ones((2, 2, 2)))
b = net.add_node(np.ones((2, 2, 2)))
# pylint: disable=pointless-statement
a[0] ^ b[1]
node = tensornetwork.contractors.optimal(net).get_final_node()
print(node.tensor)