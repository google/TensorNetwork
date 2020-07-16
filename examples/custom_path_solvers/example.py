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
import numpy as np
import tensornetwork as tn
from tensornetwork import ncon, contractors
import opt_einsum as oe
# pylint: disable=line-too-long
from tensornetwork.contractors.custom_path_solvers.nconinterface import ncon_solver
"""
An example for using`ncon_solver` to find an optimal contraction path for a
networks defined in the `ncon` syntax. Note that there are essentially three
ways to use the solver:

(i) Unrestricted search: set 'max_branch=None' to search over all possible
contraction paths in order to obtain the guaranteed optimal path. The total
search time required scales with number of tensors `N ` as: t ~ exp(N)

(ii) Restricted search: set 'max_branch' as an integer to restrict the search
to that number of the most likely paths. The total search time required scales
with number of tensors `N ` as: t ~ N*max_branch

(iii) Greedy search: set 'max_branch=1' to build a contraction path from the
sequence of locally optimal contractions (aka the greedy algorithm). The total
search time is essentially negligible: t < 0.005s
"""

# define a network (here from a 1D binary MERA algorithm)
chi = 3
chi_p = 3
u = np.random.rand(chi, chi, chi_p, chi_p)
w = np.random.rand(chi_p, chi_p, chi)
ham = np.random.rand(chi, chi, chi, chi, chi, chi)
tensors = [u, u, w, w, w, ham, u, u, w, w, w]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, -4], [11, 12, -5],
            [13, 14, -6], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], [5, 6, 16, 15],
            [8, 9, -1], [17, 16, -2], [15, 14, -3]]

t0 = time.time()
# check all contraction paths to find the optimal order
con_order, costs, is_optimal = ncon_solver(tensors, connects, max_branch=None)

# contract network using ncon
T0 = ncon(tensors, connects, con_order)
print("ncon_solver: time to contract = ", time.time() - t0)
"""
For comparison, the also show how the same network can be contracted using the
`opt_einsum` package.
"""

# combine tensors and connects lists
N = len(tensors)
comb_list = [0] * (2 * len(tensors))
for k in range(N):
  comb_list[2 * k] = tensors[k]
  comb_list[2 * k + 1] = connects[k]

# solve order and contract network using opt_einsum
t0 = time.time()
T1 = oe.contract(*comb_list, [-1, -2, -3, -4, -5, -6], optimize='branch-all')
print("opt_einsum: time to contract = ", time.time() - t0)
"""
For a final comparison, we demonstrate how the example network can be solved
for the optimal order and contracted using the node/edge API with opt_einsum
"""

# define network nodes
backend = "numpy"
iso_l = tn.Node(w, backend=backend)
iso_c = tn.Node(w, backend=backend)
iso_r = tn.Node(w, backend=backend)
iso_l_con = tn.conj(iso_l)
iso_c_con = tn.conj(iso_c)
iso_r_con = tn.conj(iso_r)
op = tn.Node(ham, backend=backend)
un_l = tn.Node(u, backend=backend)
un_l_con = tn.conj(un_l)
un_r = tn.Node(u, backend=backend)
un_r_con = tn.conj(un_r)

# define network edges
tn.connect(iso_l[0], iso_l_con[0])
tn.connect(iso_l[1], un_l[2])
tn.connect(iso_c[0], un_l[3])
tn.connect(iso_c[1], un_r[2])
tn.connect(iso_r[0], un_r[3])
tn.connect(iso_r[1], iso_r_con[1])
tn.connect(un_l[0], un_l_con[0])
tn.connect(un_l[1], op[3])
tn.connect(un_r[0], op[4])
tn.connect(un_r[1], op[5])
tn.connect(op[0], un_l_con[1])
tn.connect(op[1], un_r_con[0])
tn.connect(op[2], un_r_con[1])
tn.connect(un_l_con[2], iso_l_con[1])
tn.connect(un_l_con[3], iso_c_con[0])
tn.connect(un_r_con[2], iso_c_con[1])
tn.connect(un_r_con[3], iso_r_con[0])

# define output edges
output_edge_order = [
    iso_l_con[2], iso_c_con[2], iso_r_con[2], iso_l[2], iso_c[2], iso_r[2]
]

# solve for optimal order and contract the network
t0 = time.time()
T2 = contractors.branch(
    tn.reachable(op), output_edge_order=output_edge_order).get_tensor()
print("tn.contractors: time to contract = ", time.time() - t0)
