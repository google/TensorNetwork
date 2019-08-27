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
import tensornetwork 
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict
Tensor = Any

class FiniteMPS(tensornetwork.TensorNetwork):
    """
    An MPS class for finite systems.
    """
    def __init__(self, tensors: List[Tensor], center_position: int,
                 name: Optional[Text] = None,
                 backend: Optional[Text] = None):
        
        super().__init__(backend=backend)
        self.nodes = [self.add_node(tensors[n], name='node{}'.format(n)) 
                      for n in range(len(tensors))] #redundant?!
        for site in range(len(self.nodes) - 1):
            self.connect(self.nodes[site][2],self.nodes[site + 1][0])
        self.center_position = center_position
        
    def position(self, site: int, normalize: Optional[bool] = True) -> None:
        if site >= len(self.nodes) or site < 0:
            raise ValueError('site = {} not between values'
                             ' 0 < site < N = {}'.format(site, len(self)))
        if site == self.center_position:
            return 
        elif site > self.center_position:
            n = self.center_position
            for n in range(self.center_position, site):
                Q, R = self.split_node_qr(self.nodes[n], left_edges=[self.nodes[n][0], self.nodes[n][1]], 
                                          right_edges=[self.nodes[n][2]],
                                          left_name=self.nodes[n].name)
                Z = self.backend.norm(R.tensor)
                if normalize:
                    R.tensor /= Z
                self.nodes[n] = Q
                self.nodes[n + 1] = self.contract(R[1], name=self.nodes[n + 1].name)


            self.center_position = site
        elif site < self.center_position:
            for n in reversed(range(site + 1, self.center_position + 1)):
                R, Q = self.split_node_rq(self.nodes[n], left_edges=[self.nodes[n][0]], 
                                          right_edges=[self.nodes[n][1], self.nodes[n][2]],
                                          right_name=self.nodes[n].name)
                Z = self.backend.norm(R.tensor)                
                if normalize:
                    R.tensor /= Z
                self.nodes[n] = Q
                self.nodes[n - 1] = self.contract(R[0], name=self.nodes[n - 1].name)

            self.center_position = site
        return Z

    def transfer_operator(self, direction, tensor):
        raise NotImplementedError()
        
    def unitcell_transfer_operator(self, direction, tensor):
        net =  self.copy()
        net.add_subnetwork(net.copy())
        net.add_node()
        raise NotImplementedError()
        
    def TMeigs(self, direction):
        raise NotImplementedError()

    def canonicalize(self):
        raise NotImplementedError()
        
    def orthonormalize_left(self):
        raise NotImplementedError()
        
    def orthonormalize_right(self):
        raise NotImplementedError()
        
    def __len__(self):
        return len(self.nodes)
    
    
        
        
            
