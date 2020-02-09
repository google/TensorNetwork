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
from tensornetwork.block_tensor.block_tensor import BlockSparseTensor, reshape
from tensornetwork.block_tensor.index import Index
from tensornetwork.block_tensor.charge import U1Charge
B = 4  # possible charges on each leg can be between [0,B)
##########################################################
#####     Generate a rank 4 symmetrix tensor       #######
##########################################################

# generate random charges on each leg of the tensor
D1, D2, D3, D4 = 4, 6, 8, 10  #bond dimensions on each leg
q1 = U1Charge(np.random.randint(-B, B + 1, D1))
q2 = U1Charge(np.random.randint(-B, B + 1, D2))
q3 = U1Charge(np.random.randint(-B, B + 1, D3))
q4 = U1Charge(np.random.randint(-B, B + 1, D4))

# generate Index objects for each leg. neccessary for initialization of
# BlockSparseTensor
i1 = Index(charges=q1, flow=1)
i2 = Index(charges=q2, flow=-1)
i3 = Index(charges=q3, flow=1)
i4 = Index(charges=q4, flow=-1)

# initialize a random symmetric tensor
A = BlockSparseTensor.randn(indices=[i1, i2, i3, i4], dtype=np.complex128)
B = reshape(A, (4, 48, 10))  #creates a new tensor (copy)
shape_A = A.shape  #returns the dense shape of A
A.reshape([shape_A[0] * shape_A[1], shape_A[2],
           shape_A[3]])  #in place reshaping
A.reshape(shape_A)  #reshape back into original shape

sparse_shape = A.sparse_shape  #returns a deep copy of `A.indices`.

new_sparse_shape = [
    sparse_shape[0] * sparse_shape[1], sparse_shape[2], sparse_shape[3]
]
B = reshape(A, new_sparse_shape)  #return a  copy
B.reshape(sparse_shape)  #bring back into original shape
