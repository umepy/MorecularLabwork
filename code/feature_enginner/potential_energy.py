#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[44]:


def potential_energy():
    
    potential_energy_df = pd.read_csv('../../data/potential_energy.csv')
    print(potential_energy_df.columns)
    potential_energy = potential_energy_df[['potential_energy']].values
    return potential_energy


# In[ ]:




