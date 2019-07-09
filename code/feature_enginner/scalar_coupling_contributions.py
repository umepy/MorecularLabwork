#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

'''
scc_df : scalar_coupling_contribution_df
fc : fermi contact contribution
sd : spin-dipolar contribution
pso: paramagnetic spin-orbit contribution
dso: diamagnetic spin-orbit
'''

def get_scalar_coupling_contributions(scc_df):
    four_contributions = scc_df[['fc','sd','pso','dso']]
    
    return four_contributions, ['fermi_contact','spin_dipolar','paramagnetic_spin-orbit','diamagnetic_spin-orbit']

