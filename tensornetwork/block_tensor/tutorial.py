# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensornetwork as tn
import numpy as np
import tensornetwork.block_tensor.block_tensor as BT
import tensornetwork.block_tensor.index as IDX

B = 4  # possible charges on each leg can be between [0,B)
##########################################################
#####     Generate a rank 4 symmetrix tensor       #######
##########################################################

# generate random charges on each leg of the tensor
D1, D2, D3, D4 = 4, 6, 8, 10  #bond dimensions on each leg
q1 = np.random.randint(0, B, D1)
q2 = np.random.randint(0, B, D2)
q3 = np.random.randint(0, B, D3)
q4 = np.random.randint(0, B, D4)

# generate Index objects for each leg. neccessary for initialization of
# BlockSparseTensor
i1 = IDX.Index(charges=q1, flow=1)
i2 = IDX.Index(charges=q2, flow=-1)
i3 = IDX.Index(charges=q3, flow=1)
i4 = IDX.Index(charges=q4, flow=-1)

# initialize a random symmetric tensor
A = BT.BlockSparseTensor.randn(indices=[i1, i2, i3, i4], dtype=np.complex128)
B = BT.reshape(A, (4, 48, 10))
