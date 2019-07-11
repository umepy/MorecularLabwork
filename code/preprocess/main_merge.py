#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import time
from atom_distance import atom_distance
from scalar_coupling_contribution import get_scalar_coupling_contribution
from mulliken_charge import get_mulliken_charge
from potential_energy import get_potential_energy

def merge_features():
    start = time.time()
    train_X = []
    names = []

    # READ each features
    data, name = atom_distance()
    train_X.append(data)
    names.extend(name)

    data, name = get_potential_energy()
    train_X.append(data)
    names.extend(name)

    data, name = get_mulliken_charge()
    train_X.append(data)
    names.extend(name)

    train_X = np.hstack(train_X)
    print('Finish Preprocess:{0:.1f} sec'.format(time.time()-start))
    return train_X, names

