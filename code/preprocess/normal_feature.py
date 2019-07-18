#!/usr/bin/env python
# coding: utf-8


import pandas as pd

class normal_feature():
    def __init__(self, train_df):
        self.train_df = train_df
        
    def atom_distance(self):
        distance = pd.read_pickle('../../pkl/atom_distance.pkl')
        return distance, ['atom_distance']
    
    def get_atom_radius(self):
        atom_radius = pd.read_pickle('../../pkl/atom_radius.pkl')
        return atom_radius, ['atom_0_raidus', 'atom_1_radius']
    
    def get_atom_electron(self):
        atom_electron = pd.read_pickle('../../pkl/atom_electron.pkl')
        return atom_electron, ['atom_0_electron', 'atom_1_electron']
    
    def get_atom_weight(self):
        atom_weight = pd.read_pickle('../../pkl/atom_weight.pkl')
        return atom_weight, ['atom_0_weight', 'atom_1_weight']
    
    def get_type(self):
        types = pd.get_dummies(self.train_df['type'])
        names = types.columns
        types = types.values
        return types, names

