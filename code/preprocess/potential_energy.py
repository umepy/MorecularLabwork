#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def get_potential_energy():
    potential_energy = pd.read_pickle('../../pkl/potential_energy.pkl')
    return potential_energy, ['AD_potential_energy']

