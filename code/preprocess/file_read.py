#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

def get_df(file_name_list):
    df_list = list()
    for f in file_name_list:
        df_list.append(pd.read_csv('../../data/'+str(f)+'.csv'))
    return df_list

