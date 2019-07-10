#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

'''
mulliken_charge : マリケン電荷
'''

def get_mulliken_charge(mulliken_charge_df):
    mulliken_charge = mulliken_charge_df['mulliken_charge'].values
    mulliken_charge = mulliken_charge.reshape((len(mulliken_charge),1))
    
    return mulliken_charge, ['mulliken_charge']

