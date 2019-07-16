#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from atom_distance import atom_distance
from scalar_coupling_contribution import get_fermi_contact
from molecule_type import get_type
from file_read import get_df

'''
-filenames
train, test, structures, samplesubmission
-additional(for the molecules in train only)
dipole_moments
magnetic_shielding_tensors
mulliken_charges
potential_energy
scalar_coupling_contributions
'''

def merge_features():
    start = time.time()
    train_X = []
    names = []
    list_df = get_df(['scalar_coupling_contributions'])

    #READ each features
    data, name = atom_distance()
    train_X.append(data)
    names.extend(name)
    
    data, name = get_fermi_contact(list_df[0])
    train_X.append(data)
    names.extend(name)

    train_X = np.hstack(train_X)
    print('Finish Preprocess:{0:.1f} sec'.format(time.time()-start))
    return train_X, names

