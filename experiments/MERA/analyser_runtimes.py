from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import time
import pickle
import argparse
from sys import stdout
import datetime
date = datetime.today()

parser = argparse.ArgumentParser('bla')
parser.add_argument('filename', help='filename',type=str)
args = parser.parse_args()
        
with open(args.filename,'rb') as f:
    data = pickle.load(f)

runtimes = data['profile']
print(runtimes[16].keys())
calculated_totals={}
for chi, v1 in runtimes.items():
    keys = list(v1.keys())
    ct=np.zeros(len(v1[keys[0]]))
    for n in range(len(v1[keys[0]])):
        for k in keys:
            if k != 'total':
                ct[n] += v1[k][n]
    calculated_totals[chi] = ct
print(calculated_totals[16])
print(runtimes[16]['total'])



