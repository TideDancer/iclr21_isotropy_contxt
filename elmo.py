import h5py
import sys
import numpy as np
import pickle

f_embeds = sys.argv[1]
f_text = sys.argv[2]
ds = sys.argv[3]
layer = sys.argv[4]

h5py_file = h5py.File(f_embeds, 'r')

with open(f_text) as f:
    r = f.read().splitlines()
f.close()
length = len(r)

token_dict = {}
y = []
for i in range(length):
    print(i, '/', length)
    embeds = h5py_file.get(str(i))[int(layer)]
    keys = r[i].split(' ')
    if len(embeds) == len(keys):
        for j in range(len(embeds)):
            token_dict[keys[j]] = token_dict.get(keys[j], []) + [(embeds[j], i, j)]
            # y.append((embeds[j]))
    
pickle.dump((-1, token_dict), open('embeds/' + ds + '/elmo.layer.'+layer+'.dict', 'wb'))
