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
"""Trotter decomposition of a Hamiltonian evolution."""

import tensorflow as tf


def trotter_prepare_gates(H, step_size, num_sites, euclidean):
  """Prepare gates using 1st-order trotter decomposition.

  Currently only implemented for nearest-neighbor Hamiltonians.

  Args:
    H: List of Hamiltonian terms. Should be length num_sites-1.
    step_size: The trotter step size (a scalar).
    num_sites: The total number of sites in the system (an integer).
    euclidean: Whether the evolution is euclidean, or not (boolean).
  Returns:
    layers: A list of layers, with each layer a list of gates, one for each
      site, or `None` if no gate is applied to that site in the layer.
  """
  if not len(H) == num_sites - 1:
    raise ValueError("Number of H terms must match number of sites - 1.")

  step_size = tf.cast(step_size, tf.float64)  # must be real
  step_size = tf.cast(step_size, H[0].dtype)

  if euclidean:
    step_size = -1.0 * step_size
  else:
    step_size = 1.j * step_size

  eH = []
  for h in H:
    if len(h.shape) != 4:
      raise ValueError("H must be nearest-neighbor.")

    h_shp = tf.shape(h)
    h_r = tf.reshape(h, (h_shp[0] * h_shp[1], h_shp[2] * h_shp[3]))

    eh_r = tf.linalg.expm(step_size * h_r)
    eH.append(tf.reshape(eh_r, h_shp))

  eh_even = [None] * num_sites
  eh_odd = [None] * num_sites
  for (n, eh) in enumerate(eH):
    if n % 2 == 0:
      eh_even[n] = eh
    else:
      eh_odd[n] = eh

  return [eh_even, eh_odd]
