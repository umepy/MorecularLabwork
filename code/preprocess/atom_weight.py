#!/usr/bin/env python
# coding: utf-8

import pandas as pd

'''
shape : (len(train_df),2)

'''

def get_atom_weight():
    atom_weight = pd.read_pickle('../../pkl/atom_weight.pkl')
    return atom_weight, ['atom_0_weight', 'atom_1_weight']

def atom_weight_product():
    atom_weight = pd.read_pickle('../../pkl/atom_weight.pkl')
    atom_weight_product = atom_weight[:,0] * atom_weight[:,1]
    atom_weight_product = atom_weight_product.reshape(len(atom_weight_product),1)
    return atom_weight_product, ['atom_weight_product']

