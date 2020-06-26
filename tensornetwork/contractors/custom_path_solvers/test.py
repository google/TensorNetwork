# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:58:59 2020

@author: gevenbly3
"""
from yapf.yapflib.yapf_api import FormatFile
FormatFile(
    "tensornetwork/contractors/custom_path_solvers/nconinterface.py",
    in_place=True,
    style_config='C:\\Users\\gevenbly3\\Documents\\GitHub\\TensorNetwork\\.style.yapf'
)
FormatFile(
    "tensornetwork/contractors/custom_path_solvers/pathsolvers.py",
    in_place=True)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from typing import List, Union, Any, Tuple, Type, Optional, Sequence
