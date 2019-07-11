#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def atom_distance():
    distance = pd.read_pickle('../../pkl/atom_distance.pkl')
    return distance, ['atom_distance']

