#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

'''
mulliken_charge : マリケン電荷
shape  : (len(train_df), 2)
columns: [atom_index_0's mulliken_charge, atom_index_1's mulliken_charge] 

'''

def get_mulliken_charge():
    mulliken_charge = pd.read_pickle('../../pkl/mulliken_charge.pkl')
    return mulliken_charge, ['AD_atom_0_mulliken_charge', 'AD_atom_1_mulliken_charge']

