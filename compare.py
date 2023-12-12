import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_dft = pd.read_pickle('df_dft.pkl')
df_casscf = pd.read_pickle('df_casscf.pkl')

print(df_dft.columns)
print(df_casscf.columns)


# df columns: file, function, basis, osc_prod, wavelength_prod, osc_react, wavelength_react, energy_prod, energy_react, energy_ts, tbr_energy, storage_energy 
true_osc_prod = df_casscf['osc_prod']
true_osc_react = df_casscf['osc_react']
true_wavelength_prod = df_casscf['wavelength_prod']
true_wavelength_react = df_casscf['wavelength_react']
true_energy_prod = df_casscf['energy_prod']
true_energy_react = df_casscf['energy_react']
true_energy_ts = df_casscf['energy_ts']
true_tbr_energy = df_casscf['tbr_energy']
true_storage_energy = df_casscf['storage_energy']
