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


def test_constructor(backend):
  psi_tensor = np.random.rand(2, 2)
  psi_node = tn.Node(psi_tensor, backend=backend)

  op = qu.quantum_constructor([psi_node[0]], [psi_node[1]])
  assert not op.is_scalar()
  assert not op.is_vector()
  assert not op.is_adjoint_vector()
  assert len(op.out_edges) == 1
  assert len(op.in_edges) == 1
  assert op.out_edges[0] is psi_node[0]
  assert op.in_edges[0] is psi_node[1]

  op = qu.quantum_constructor([psi_node[0], psi_node[1]], [])
  assert not op.is_scalar()
  assert op.is_vector()
  assert not op.is_adjoint_vector()
  assert len(op.out_edges) == 2
  assert len(op.in_edges) == 0
  assert op.out_edges[0] is psi_node[0]
  assert op.out_edges[1] is psi_node[1]

  op = qu.quantum_constructor([], [psi_node[0], psi_node[1]])
  assert not op.is_scalar()
  assert not op.is_vector()
  assert op.is_adjoint_vector()
  assert len(op.out_edges) == 0
  assert len(op.in_edges) == 2
  assert op.in_edges[0] is psi_node[0]
  assert op.in_edges[1] is psi_node[1]

  with pytest.raises(ValueError):
    op = qu.quantum_constructor([], [], [psi_node])

  _ = psi_node[0] ^ psi_node[1]
  op = qu.quantum_constructor([], [], [psi_node])
  assert op.is_scalar()
  assert not op.is_vector()
  assert not op.is_adjoint_vector()
  assert len(op.out_edges) == 0
  assert len(op.in_edges) == 0


def test_checks(backend):
  node1 = tn.Node(np.random.rand(2, 2), backend=backend)
  node2 = tn.Node(np.random.rand(2, 2), backend=backend)
  _ = node1[1] ^ node2[0]

  # extra dangling edges must be explicitly ignored
  with pytest.raises(ValueError):
    _ = qu.QuVector([node1[0]])

  # correctly ignore the extra edge
  _ = qu.QuVector([node1[0]], ignore_edges=[node2[1]])

  # in/out edges must be dangling
  with pytest.raises(ValueError):
    _ = qu.QuVector([node1[0], node1[1], node2[1]])


def test_from_tensor(backend):
  psi_tensor = np.random.rand(2, 2)

  op = qu.QuOperator.from_tensor(psi_tensor, [0], [1], backend=backend)
  assert not op.is_scalar()
  assert not op.is_vector()
  assert not op.is_adjoint_vector()
  np.testing.assert_almost_equal(op.eval(), psi_tensor)

  op = qu.QuVector.from_tensor(psi_tensor, [0, 1], backend=backend)
  assert not op.is_scalar()
  assert op.is_vector()
  assert not op.is_adjoint_vector()
  np.testing.assert_almost_equal(op.eval(), psi_tensor)

  op = qu.QuAdjointVector.from_tensor(psi_tensor, [0, 1], backend=backend)
  assert not op.is_scalar()
  assert not op.is_vector()
  assert op.is_adjoint_vector()
  np.testing.assert_almost_equal(op.eval(), psi_tensor)

  op = qu.QuScalar.from_tensor(1.0, backend=backend)
  assert op.is_scalar()
  assert not op.is_vector()
  assert not op.is_adjoint_vector()
  assert op.eval() == 1.0


def test_identity(backend):
  E = qu.identity((2, 3, 4), backend=backend)
  for n in E.nodes:
    assert isinstance(n, tn.CopyNode)
  twentyfour = E.trace()
  for n in twentyfour.nodes:
    assert isinstance(n, tn.CopyNode)
  assert twentyfour.eval() == 24

  tensor = np.random.rand(2, 2)
  psi = qu.QuVector.from_tensor(tensor, backend=backend)
  E = qu.identity((2, 2), backend=backend)
  np.testing.assert_allclose((E @ psi).eval(), psi.eval())

  np.testing.assert_allclose((psi.adjoint() @ E @ psi).eval(),
                             psi.norm().eval())

  op = qu.QuOperator.from_tensor(tensor, [0], [1], backend=backend)
  op_I = op.tensor_product(E)
  op_times_4 = op_I.partial_trace([1, 2])
  np.testing.assert_allclose(op_times_4.eval(), 4 * op.eval())


def test_tensor_product(backend):
  psi = qu.QuVector.from_tensor(np.random.rand(2, 2), backend=backend)
  psi_psi = psi.tensor_product(psi)
  assert len(psi_psi.subsystem_edges) == 4
  np.testing.assert_almost_equal(psi_psi.norm().eval(), psi.norm().eval()**2)


def test_matmul(backend):
  mat = np.random.rand(2, 2)
  op = qu.QuOperator.from_tensor(mat, [0], [1], backend=backend)
  res = (op @ op).eval()
  np.testing.assert_allclose(res, mat @ mat)


def test_mul(backend):
  mat = np.eye(2)
  scal = np.float64(0.5)
  op = qu.QuOperator.from_tensor(mat, [0], [1], backend=backend)
  scal_op = qu.QuScalar.from_tensor(scal, backend=backend)

  res = (op * scal_op).eval()
  np.testing.assert_allclose(res, mat * 0.5)

  res = (scal_op * op).eval()
  np.testing.assert_allclose(res, mat * 0.5)

  res = (scal_op * scal_op).eval()
  np.testing.assert_almost_equal(res, 0.25)

  res = (op * np.float64(0.5)).eval()
  np.testing.assert_allclose(res, mat * 0.5)

  res = (np.float64(0.5) * op).eval()
  np.testing.assert_allclose(res, mat * 0.5)

  with pytest.raises(ValueError):
    _ = (op * op)

  with pytest.raises(ValueError):
    _ = (op * mat)


def test_expectations(backend):
  if backend == 'pytorch':
    psi_tensor = np.random.rand(2, 2, 2)
    op_tensor = np.random.rand(2, 2)
  else:
    psi_tensor = np.random.rand(2, 2, 2) + 1.j * np.random.rand(2, 2, 2)
    op_tensor = np.random.rand(2, 2) + 1.j * np.random.rand(2, 2)

  psi = qu.QuVector.from_tensor(psi_tensor, backend=backend)
  op = qu.QuOperator.from_tensor(op_tensor, [0], [1], backend=backend)

  op_3 = op.tensor_product(
      qu.identity((2, 2), backend=backend, dtype=psi_tensor.dtype))
  res1 = (psi.adjoint() @ op_3 @ psi).eval()

  rho_1 = psi.reduced_density([1, 2])  # trace out sites 2 and 3
  res2 = (op @ rho_1).trace().eval()

  np.testing.assert_almost_equal(res1, res2)


def test_projector(backend):
  psi_tensor = np.random.rand(2, 2)
  psi_tensor /= np.linalg.norm(psi_tensor)
  psi = qu.QuVector.from_tensor(psi_tensor, backend=backend)
  P = psi.projector()
  np.testing.assert_allclose((P @ psi).eval(), psi_tensor)

  np.testing.assert_allclose((P @ P).eval(), P.eval())
