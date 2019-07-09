#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../data/"))

import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time
# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


jobs = 10


# In[3]:


train_df = pd.read_csv('../data/train.csv')
scc_df = pd.read_csv('../data/scalar_coupling_contributions.csv')


# In[4]:


train_df.head(10)


# In[5]:


scc_df.head(10)


# In[6]:


train_df.shape, scc_df.shape


# In[8]:


scc_arr = scc_df[['fc','sd','pso','dso']].values
scc_arr.shape


# In[9]:


def contribution():
    scc_df = pd.read_csv('../data/scalar_coupling_contributions.csv')
    scc_contribution = scc_df[['fc','sd','pso','dso']].values
    return scc_contribution


# In[10]:


contribution()

