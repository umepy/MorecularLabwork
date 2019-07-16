#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def get_type(train_df):
    types = pd.get_dummies(train_df['type'])
    names = types.columns
    types = types.values
    return types, names
