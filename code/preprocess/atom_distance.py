#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def atom_distance():
    distance = pd.read_pickle('../../pkl/atom_distance.pkl')
    return distance, ['atom_distance']

def atom_distance_p():
    distance = pd.read_pickle('../../pkl/atom_distance_p.pkl')
    return distance, ['atom_distance_p']

