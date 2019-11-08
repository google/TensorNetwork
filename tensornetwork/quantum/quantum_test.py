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

import pytest
import numpy as np
import tensornetwork as tn
import quantum as qu


def test_constructor():
  psi_tensor = np.random.rand(2,2)
  psi_node = tn.Node(psi_tensor)

  op = qu.quantum_constructor([psi_node[0]], [psi_node[1]])
  assert not op.is_scalar
  assert not op.is_vector
  assert not op.is_adjoint_vector

  op = qu.quantum_constructor([psi_node[0], psi_node[1]], [])
  assert not op.is_scalar
  assert op.is_vector
  assert not op.is_adjoint_vector

  op = qu.quantum_constructor([], [psi_node[0], psi_node[1]])
  assert not op.is_scalar
  assert not op.is_vector
  assert op.is_adjoint_vector

def test_from_tensor():
  psi_tensor = np.random.rand(2,2)

  op = qu.QuOperator.from_tensor(psi_tensor, [0], [1])
  assert not op.is_scalar
  assert not op.is_vector
  assert not op.is_adjoint_vector

  op = qu.QuVector.from_tensor(psi_tensor)
  assert not op.is_scalar
  assert op.is_vector
  assert not op.is_adjoint_vector

  op = qu.QuAdjointVector.from_tensor(psi_tensor)
  assert not op.is_scalar
  assert not op.is_vector
  assert op.is_adjoint_vector