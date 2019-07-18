#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import normal_feature
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
    list_df = get_df(['train', 'scalar_coupling_contributions'])
    get_feature = normal_feature.normal_feature(list_df[0])
    

    #READ each features
    data, name = get_feature.atom_distance()
    train_X.append(data)
    names.extend(name)
    
#     data, name = get_feature.get_atom_weight()
#     train_X.append(data)
#     names.extend(name)

#     data, name = get_feature.get_fermi_contact(list_df[1])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_feature.get_spin_dipolar(list_df[1])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_feature.get_p_spin_orbit(list_df[1])
#     train_X.append(data)
#     names.extend(name)
    
#     data, name = get_feature.get_d_spin_orbit(list_df[1])
#     train_X.append(data)
#     names.extend(name)
    
    train_X = np.hstack(train_X)
    print('Finish Preprocess:{0:.1f} sec'.format(time.time()-start))
    return train_X, names

