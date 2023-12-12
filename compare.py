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


# Create a 2d array with basis and function with absolute error for each property
basis = df_dft['basis'].unique()
function = df_dft['function'].unique()
abs_error_osc_prod = np.zeros((len(basis), len(function)))
abs_error_osc_react = np.zeros((len(basis), len(function)))
abs_error_wavelength_prod = np.zeros((len(basis), len(function)))
abs_error_wavelength_react = np.zeros((len(basis), len(function)))
abs_error_energy_prod = np.zeros((len(basis), len(function)))
abs_error_energy_react = np.zeros((len(basis), len(function)))
abs_error_energy_ts = np.zeros((len(basis), len(function)))
abs_error_tbr_energy = np.zeros((len(basis), len(function)))
abs_error_storage_energy = np.zeros((len(basis), len(function)))

for i in range(len(basis)):
    for j in range(len(function)):
        df_temp = df_dft[(df_dft['basis'] == basis[i]) & (df_dft['function'] == function[j])]
        abs_error_osc_prod[i][j] = np.mean(np.abs(df_temp['osc_prod'] - true_osc_prod))
        abs_error_osc_react[i][j] = np.mean(np.abs(df_temp['osc_react'] - true_osc_react))
        abs_error_wavelength_prod[i][j] = np.mean(np.abs(df_temp['wavelength_prod'] - true_wavelength_prod))
        abs_error_wavelength_react[i][j] = np.mean(np.abs(df_temp['wavelength_react'] - true_wavelength_react))
        abs_error_energy_prod[i][j] = np.mean(np.abs(df_temp['energy_prod'] - true_energy_prod))
        abs_error_energy_react[i][j] = np.mean(np.abs(df_temp['energy_react'] - true_energy_react))
        abs_error_energy_ts[i][j] = np.mean(np.abs(df_temp['energy_ts'] - true_energy_ts))
        abs_error_tbr_energy[i][j] = np.mean(np.abs(df_temp['tbr_energy'] - true_tbr_energy))
        abs_error_storage_energy[i][j] = np.mean(np.abs(df_temp['storage_energy'] - true_storage_energy))

# Plot the absolute error for each property as a heatmap
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
sns.heatmap(abs_error_osc_prod, annot=True, ax=ax[0,0])
sns.heatmap(abs_error_osc_react, annot=True, ax=ax[0,1])
sns.heatmap(abs_error_wavelength_prod, annot=True, ax=ax[0,2])
sns.heatmap(abs_error_wavelength_react, annot=True, ax=ax[1,0])
sns.heatmap(abs_error_energy_prod, annot=True, ax=ax[1,1])
sns.heatmap(abs_error_energy_react, annot=True, ax=ax[1,2])
sns.heatmap(abs_error_energy_ts, annot=True, ax=ax[2,0])
sns.heatmap(abs_error_tbr_energy, annot=True, ax=ax[2,1])
sns.heatmap(abs_error_storage_energy, annot=True, ax=ax[2,2])

ax[0,0].set_title('Oscillator Strength (Prod)')
ax[0,1].set_title('Oscillator Strength (React)')
ax[0,2].set_title('Wavelength (Prod)')
ax[1,0].set_title('Wavelength (React)')
ax[1,1].set_title('Energy (Prod)')
ax[1,2].set_title('Energy (React)')
ax[2,0].set_title('Energy (TS)')
ax[2,1].set_title('TBR Energy')
ax[2,2].set_title('Storage Energy')

ax[0,0].set_xticklabels(function)
ax[0,1].set_xticklabels(function)
ax[0,2].set_xticklabels(function)
ax[1,0].set_xticklabels(function)
ax[1,1].set_xticklabels(function)
ax[1,2].set_xticklabels(function)
ax[2,0].set_xticklabels(function)
ax[2,1].set_xticklabels(function)
ax[2,2].set_xticklabels(function)

ax[0,0].set_yticklabels(basis)
ax[0,1].set_yticklabels(basis)
ax[0,2].set_yticklabels(basis)
ax[1,0].set_yticklabels(basis)
ax[1,1].set_yticklabels(basis)
ax[1,2].set_yticklabels(basis)
ax[2,0].set_yticklabels(basis)
ax[2,1].set_yticklabels(basis)
ax[2,2].set_yticklabels(basis)

plt.tight_layout()
plt.savefig('compare.png')
