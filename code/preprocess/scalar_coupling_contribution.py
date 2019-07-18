#!/usr/bin/env python
# coding: utf-8

import pandas as pd

'''
scc_df : scalar_coupling_contribution_df
fc : fermi contact contribution
sd : spin-dipolar contribution
pso: paramagnetic spin-orbit contribution
dso: diamagnetic spin-orbit
'''

def get_fermi_contact(scc_df):
    fermi_contact = scc_df['fc'].values
    fermi_contact = fermi_contact.reshape(len(fermi_contact),1)
    return fermi_contact, ['AD_fermi_contact']

def get_spin_dipolar(scc_df):
    spin_dipolar = scc_df['sd'].values
    spin_dipolar = spin_dipolar.reshape(len(spin_dipolar),1)
    return spin_dipolar, ['AD_spin_dipolar']

def get_p_spin_orbit(scc_df):
    p_spin_orbit = scc_df['pso'].values
    p_spin_orbit = p_spin_orbit.reshape(len(p_spin_orbit),1)
    return p_spin_orbit, ['AD_paramagnetic_spin_orbit']

def get_d_spin_orbit(scc_df):
    d_spin_orbit = scc_df['dso'].values
    d_spin_orbit = d_spin_orbit.reshape(len(d_spin_orbit),1)
    return d_spin_orbit, ['AD_diamagnetic_spin_orbit']

