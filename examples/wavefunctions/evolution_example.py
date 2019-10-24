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
"""Trotter evolution of exact wavefunctions: Example script."""

import tensorflow as tf

from examples.wavefunctions import wavefunctions


def ising_hamiltonian(N, dtype):
  X = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
  Z = tf.convert_to_tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
  I = tf.eye(2, dtype=dtype)
  h = -tf.tensordot(X, X, axes=0) - tf.tensordot(Z, I, axes=0)
  h_last = h - tf.tensordot(I, Z, axes=0)
  h = tf.transpose(h, (0, 2, 1, 3))
  h_last = tf.transpose(h_last, (0, 2, 1, 3))
  H = [h] * (N - 2) + [h_last]
  return H


def random_state(N, d, dtype):
  psi = tf.cast(tf.random.uniform([d for n in range(N)]), dtype)
  psi = tf.divide(psi, tf.norm(psi))
  return psi


def callback(psi, t, i):
  print(i,
        tf.norm(psi).numpy().real,
        wavefunctions.expval(psi, X, 0).numpy().real)


if __name__ == "__main__":
  N = 16
  dtype = tf.complex128
  build_graph = True

  dt = 0.1
  num_steps = 100
  euclidean_evolution = False

  print("----------------------------------------------------")
  print("Evolving a random state by the Ising Hamiltonian.")
  print("----------------------------------------------------")
  print("System size:", N)
  print("Trotter step size:", dt)
  print("Euclidean?:", euclidean_evolution)

  X = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
  H = ising_hamiltonian(N, dtype)
  psi = random_state(N, 2, dtype)

  if build_graph:
    f = wavefunctions.evolve_trotter_defun
  else:
    f = wavefunctions.evolve_trotter

  print("----------------------------------------------------")
  print("step\tnorm\t<X_0>")
  print("----------------------------------------------------")
  psi_t, t = f(
      psi, H, dt, num_steps, euclidean=euclidean_evolution, callback=callback)

  print("Final norm:", tf.norm(psi_t).numpy().real)
  print("<psi | psi_t>:", wavefunctions.inner(psi, psi_t).numpy())
