# Copyright 2020 The TensorNetwork Authors
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

"""Definition of classes FusionTree and Su2Tensor. (Very basic at present, only handles 2- and 3- index tensors)"""
import numpy as np

class FusionTree:
    """
    Class to encapsulate a FusionTree. Currently used only for 2-index and 3-index Su2Tensors. In these cases FusionTree
    is essentially a wrapper on a list of charges (2 or 3 charges)
    """
    def __init__(self, charges):
        self.charges = charges # charges is simply a tuple for now. (Should be replaced by a decorated tree in future)
    def __hash__(self):
        """
        To use objects of FusionTree as a key in a dictionary
        """
        return hash(self.charges)

    def __eq__(self, other):
        """
        To use objects of FusionTree as a key in a dictionary.
        """
        return self.charges == other.charges

    def __ne__(self, other):
        """
        To use objects of FusionTree as a key in a dictionary. Not strictly necessary, but to avoid having
        both x==y and x!=y True at the same time.
        """
        return not(self == other)

    def __getitem__(self, key):
        """
        Override the built-in [] operator, so we can access elements of charges directly using the object
        Currently, the class FusionTree is a wrapper to a list charges.
        """
        return self.charges[key]

class Su2Tensor:
    """
    Only for 2- and 3-index SU(2) tensors for now! Currently essentially a wrapper for a dict: {FusionTree,block}
    """
    def __init__(
            self, indices: list):
        self.blocks = dict() # blocks is a list indexed by keys which are FusionTrees
        self.indices = indices

    def addblock(
            self, fusiontree: FusionTree, block: np.array):
        """
        Adds a block indexed by the given fusiontree.
        """
        if fusiontree in self.blocks:
            self.blocks[fusiontree] += block
        else:
            self.blocks[fusiontree] = block

    def getblock(
            self, fusiontree: FusionTree):
        """
        Retrieves a block indexed by the given fusiontree.
        """
        if fusiontree in self.blocks:
            return self.blocks[fusiontree]
        else:
            raise KeyError('Fusion tree not found in the tensor.')

    def getAllBlocks(self):
        """
        Retrieves the whole block dictionary.
        """
        return self.blocks

    def getIndices(self):
        """
        Retrieves all the indices of the tensor.
        """
        return self.indices
