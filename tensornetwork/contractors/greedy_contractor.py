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
"""Greedy Contraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Tuple
from tensornetwork import network
from tensornetwork import network_components

def greedy(net: network.TensorNetwork) -> network.TensorNetwork:
  """Contract the lowest cost pair of nodes first.
  
  Args:
    net: The TensorNetwork to contract
  Returns:
    The contracted TensorNetwork.
  """
  # First, contract all of the trace edges.
  edges = net.get_all_nondangling()
  for edge in edges:
    if edge in net and edge.is_trace():
      net.contract_parallel(edge)
  edges = net.flatten_all_edges()


def _siftdown(heap: List[Tuple[int, network_components.Node]], startpos, pos):
  newitem = heap[pos]
  # Follow the path to the root, moving parents down until finding a place
  # newitem fits.
  while pos > startpos:
    parentpos = (pos - 1) >> 1
    parent = heap[parentpos]
    if newitem < parent:
      heap[pos] = parent
      pos = parentpos
      continue
    break
  heap[pos] = newitem


def _siftup(heap, pos):
  endpos = len(heap)
  startpos = pos
  newitem = heap[pos]
  # Bubble up the smaller child until hitting a leaf.
  childpos = 2 * pos + 1  # Leftmost child position.
  while childpos < endpos:
    # Set childpos to index of smaller child.
    rightpos = childpos + 1
    if rightpos < endpos and not heap[childpos] < heap[rightpos]:
      childpos = rightpos
    # Move the smaller child up.
    heap[pos] = heap[childpos]
    pos = childpos
    childpos = 2 * pos + 1
  # The leaf at pos is empty now. Put newitem there, and bubble it up
  # to its final resting place (by sifting its parents down).
  heap[pos] = newitem
  _siftdown(heap, startpos, pos)
