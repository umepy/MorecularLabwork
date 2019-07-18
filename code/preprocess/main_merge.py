#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from atom_distance import atom_distance
from atom_weight import get_atom_weight
from atom_weight import atom_weight_product
from scalar_coupling_contribution import get_fermi_contact
from scalar_coupling_contribution import get_spin_dipolar
from scalar_coupling_contribution import get_p_spin_orbit
from scalar_coupling_contribution import get_d_spin_orbit
from mulliken_charge import get_mulliken_charge
from potential_energy import get_potential_energy
from file_read import get_df

'''
get_df() : read file and return dataframe

-filenames-
train, test, structures, samplesubmission
-additional(for the molecules in train only)-
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
    
    data, name = get_atom_weight()
    train_X.append(data)
    names.extend(name)

#     data, name = get_fermi_contact(list_df[0])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_spin_dipolar(list_df[0])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_p_spin_orbit(list_df[0])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_d_spin_orbit(list_df[0])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_mulliken_charge()
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_potential_energy()
#     train_X.append(data)
#     names.extend(name)
    
    train_X = np.hstack(train_X)
    print('Finish Preprocess:{0:.1f} sec'.format(time.time()-start))
    return train_X, names

