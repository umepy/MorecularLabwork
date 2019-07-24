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
    
    def get_atom_x(self):
        atom_x = pd.read_pickle('../../pkl/atomic_x.pkl')
        return atom_x, ['atom_0_x', 'atom_1_x']
    
    def get_atom_y(self):
        atom_y = pd.read_pickle('../../pkl/atomic_y.pkl')
        return atom_y, ['atom_0_y', 'atom_1_y']
    
    def get_atom_z(self):
        atom_z = pd.read_pickle('../../pkl/atomic_z.pkl')
        return atom_z, ['atom_0_z', 'atom_1_z']
    
    def get_atom_gyromagnetic_ratio(self):
        atom_gyromagnetic_ratio = pd.read_pickle('../../pkl/atomic_ratio.pkl')
        return atom_gyromagnetic_ratio, ['atom_0_gyromagnetic_ratio', 'atom_1_gyromagnetic_ratio']
    
    def get_atom_angle(self):
        atom_angle = pd.read_pickle('../../pkl/angle.pkl')
        return atom_angle, ['atom_0_angle', 'atom_1_angle']
    
    def get_atom_angle_diff(self):
        atom_angle_diff = pd.read_pickle('../../pkl/angle_diff.pkl')
        return atom_angle_diff, ['angle_diff']
    
    def get_atom_x_diff(self):
        atom_x_diff = pd.read_pickle('../../pkl/x_diff.pkl')
        return atom_x_diff, ['x_diff']
    
    def get_atom_y_diff(self):
        atom_y_diff = pd.read_pickle('../../pkl/y_diff.pkl')
        return atom_y_diff, ['y_diff']
    
    def get_atom_z_diff(self):
        atom_z_diff = pd.read_pickle('../../pkl/z_diff.pkl')
        return atom_z_diff, ['z_diff']
    
    def get_atom_num(self):
        each_atom_num = pd.read_pickle('../../pkl/atom_num.pkl')
        return each_atom_num, ['C', 'H', 'N', 'O', 'F']
    
    def get_type(self):
        types = pd.get_dummies(self.train_df['type'])
        names = types.columns
        types = types.values
        return types, names

