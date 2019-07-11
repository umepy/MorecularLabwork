#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

'''
mulliken_charge : マリケン電荷

'''

def get_mulliken_charge():
    mulliken_charge = pd.read_pickle('../../tmp_data/mulliken_charge.pkl')
    # shape : (len(train_df), 2)
    # columns: [atom_index_0's mulliken_charge, atom_index_1's mulliken_charge] 
    return mulliken_charge, ['mulliken_charge']

