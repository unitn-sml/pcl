#!/usr/bin/env python3

import sys
import pickle
import numpy as np

inpaths, outpath = sys.argv[2:], sys.argv[1]

data = []
for path in inpaths:
    with open(path, 'rb') as fp:
        data.append(pickle.load(fp))

args = data[0]['args']
traces = data[0]['traces']
print(np.array(traces).shape)
for datum in data[1:]:
    traces += datum['traces']
    print(np.array(traces).shape, np.array(datum['traces']).shape)

with open(outpath, 'wb') as fp:
    pickle.dump({'args': args, 'traces': traces}, fp)

print('done')
