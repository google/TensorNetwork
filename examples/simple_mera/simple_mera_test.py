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

import time
import pytest
import jax
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import jax.random
import tensornetwork as tn
import simple_mera


def test_descend(random_tensors):
  h, s, iso, dis = random_tensors
  s = simple_mera.descend(h, s, iso, dis)
  assert len(s.shape) == 6
  D = s.shape[0]
  smat = np.reshape(s, [D**3] * 2)
  assert np.isclose(np.trace(smat), 1.0)
  assert np.isclose(np.linalg.norm(smat - np.conj(np.transpose(smat))), 0.0)
  spec, _ = np.linalg.eigh(smat)
  assert np.alltrue(spec >= 0.0)


def test_ascend(random_tensors):
  h, s, iso, dis = random_tensors
  h = simple_mera.ascend(h, s, iso, dis)
  assert len(h.shape) == 6
  D = h.shape[0]
  hmat = np.reshape(h, [D**3] * 2)
  norm = np.linalg.norm(hmat - np.conj(np.transpose(hmat)))
  assert np.isclose(norm, 0.0)


def test_energy(wavelet_tensors):
  h, iso, dis = wavelet_tensors
  s = np.reshape(np.eye(2**3) / 2**3, [2] * 6)
  for _ in range(20):
    s = simple_mera.descend(h, s, iso, dis)
  en = np.trace(np.reshape(s, [2**3, -1]) @ np.reshape(h, [2**3, -1]))
  assert np.isclose(en, -1.242, rtol=1e-3, atol=1e-3)
  en = simple_mera.binary_mera_energy(h, s, iso, dis)
  assert np.isclose(en, -1.242, rtol=1e-3, atol=1e-3)


def test_opt(wavelet_tensors):
  h, iso, dis = wavelet_tensors
  s = np.reshape(np.eye(2**3) / 2**3, [2] * 6)
  for _ in range(20):
    s = simple_mera.descend(h, s, iso, dis)
  s, iso, dis = simple_mera.optimize_linear(h, s, iso, dis, 100)
  en = np.trace(np.reshape(s, [2**3, -1]) @ np.reshape(h, [2**3, -1]))
  assert en < -1.25


@pytest.fixture(params=[2, 3])
def random_tensors(request):
  D = request.param
  key = jax.random.PRNGKey(0)

  h = jax.random.normal(key, shape=[D**3] * 2)
  h = 0.5 * (h + np.conj(np.transpose(h)))
  h = np.reshape(h, [D] * 6)

  s = jax.random.normal(key, shape=[D**3] * 2)
  s = s @ np.conj(np.transpose(s))
  s /= np.trace(s)
  s = np.reshape(s, [D] * 6)

  a = jax.random.normal(key, shape=[D**2] * 2)
  u, _, vh = np.linalg.svd(a)
  dis = np.reshape(u, [D] * 4)
  iso = np.reshape(vh, [D] * 4)[:, :, :, 0]

  return tuple(x.astype(np.complex128) for x in (h, s, iso, dis))


@pytest.fixture
def wavelet_tensors(request):
  """Returns the Hamiltonian and MERA tensors for the D=2 wavelet MERA.

  From Evenbly & White, Phys. Rev. Lett. 116, 140403 (2016).
  """
  D = 2
  h = simple_mera.ham_ising()

  E = np.array([[1, 0], [0, 1]])
  X = np.array([[0, 1], [1, 0]])
  Y = np.array([[0, -1j], [1j, 0]])
  Z = np.array([[1, 0], [0, -1]])

  wmat_un = np.real((np.sqrt(3) + np.sqrt(2)) / 4 * np.kron(E, E) +
                    (np.sqrt(3) - np.sqrt(2)) / 4 * np.kron(Z, Z) + 1.j *
                    (1 + np.sqrt(2)) / 4 * np.kron(X, Y) + 1.j *
                    (1 - np.sqrt(2)) / 4 * np.kron(Y, X))

  umat = np.real((np.sqrt(3) + 2) / 4 * np.kron(E, E) +
                 (np.sqrt(3) - 2) / 4 * np.kron(Z, Z) +
                 1.j / 4 * np.kron(X, Y) + 1.j / 4 * np.kron(Y, X))

  w = np.reshape(wmat_un, (D, D, D, D))[:, 0, :, :]
  u = np.reshape(umat, (D, D, D, D))

  w = np.transpose(w, [1, 2, 0])
  u = np.transpose(u, [2, 3, 0, 1])

  return tuple(x.astype(np.complex128) for x in (h, w, u))
