import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# print all of columns in pandas dataframe
pd.set_option('display.max_columns', None)

# Read in the dataframes
df_dft = pd.read_pickle('/groups/kemi/obel/azobenzene/compchem/comparison/dft_results.pkl')
df_casscf = pd.read_pickle('/groups/kemi/obel/azobenzene/compchem/comparison/casscf_results.pkl')

# print(df_dft.columns)
# print(df_casscf.columns)


# df columns: file, function, basis, osc_prod, wavelength_prod, osc_react, wavelength_react, energy_prod, energy_react, energy_ts, tbr_energy, storage_energy 
true_osc_prod = df_casscf['osc_prod'].values[0]
true_osc_react = df_casscf['osc_react'].values[0]
true_wavelength_prod = df_casscf['wavelength_prod'].values[0]
true_wavelength_react = df_casscf['wavelength_react'].values[0]
true_energy_prod = df_casscf['energy_prod'].values[0]
true_energy_react = df_casscf['energy_react'].values[0]
true_energy_ts = df_casscf['energy_ts'].values[0]
true_tbr_energy = df_casscf['tbr_energy'].values[0]
true_storage_energy = df_casscf['storage_energy'].values[0]

print(true_osc_prod, true_osc_react, true_wavelength_prod, true_wavelength_react, true_energy_prod, true_energy_react, true_energy_ts, true_tbr_energy, true_storage_energy)
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
        print(basis[i], function[j])
        df_temp = df_dft[(df_dft['basis'] == basis[i]) & (df_dft['function'] == function[j])]
        
        # Assign the value for each property with error catch
        if len(df_temp['osc_prod'][0]) == 0:
            abs_error_osc_prod[i][j] = -1
        else:
            abs_error_osc_prod[i][j] = np.mean(np.abs(df_temp['osc_prod'][0][0] - true_osc_prod))
        if len(df_temp['osc_react']) == 0:
            abs_error_osc_react[i][j] = -1
        else:
            abs_error_osc_react[i][j] = np.mean(np.abs(df_temp['osc_react'][0] - true_osc_react))
        if len(df_temp['wavelength_prod'][0]) == 0:
            abs_error_wavelength_prod[i][j] = -1
        else:
            abs_error_wavelength_prod[i][j] = np.mean(np.abs(df_temp['wavelength_prod'][0][0] - true_wavelength_prod))
        if len(df_temp['wavelength_react']) == 0:
            abs_error_wavelength_react[i][j] = -1
        else:
            abs_error_wavelength_react[i][j] = np.mean(np.abs(df_temp['wavelength_react'][0] - true_wavelength_react))

        # print(df_temp['energy_react'],df_temp['storage_energy'], df_temp['tbr_energy'])
        abs_error_energy_prod[i][j] = np.abs(df_temp['energy_prod'] - true_energy_prod)
        abs_error_energy_react[i][j] = np.abs(df_temp['energy_react'] - true_energy_react)
        abs_error_energy_ts[i][j] = np.abs(df_temp['energy_ts'] - true_energy_ts)
        abs_error_tbr_energy[i][j] = np.abs(df_temp['tbr_energy'] - true_tbr_energy)
        abs_error_storage_energy[i][j] = np.abs(df_temp['storage_energy'] - true_storage_energy)


# All examined properties
properties = ['osc_prod', 'osc_react', 'wavelength_prod', 'wavelength_react', 'energy_prod', 'energy_react', 'energy_ts', 'tbr_energy', 'storage_energy']
# Plot the absolute error for each property as a seperate heatmap  
# Handle the error catch for the properties that are not NaN
for i in properties:
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(eval('abs_error_'+i), annot=True, ax=ax, xticklabels=function, yticklabels=basis, cmap='viridis', fmt='.2f')
    ax.set_title('Absolute Error for '+i)
    ax.set_xlabel('Functional')
    ax.set_ylabel('Basis')
    plt.savefig('abs_error_'+i+'.png', dpi=300)
    plt.close()


#Save the absolute error dataframes as csv files
df_abs_error_osc_prod = pd.DataFrame(abs_error_osc_prod, index=basis, columns=function)
df_abs_error_osc_prod.to_csv('abs_error_osc_prod.csv')
df_abs_error_osc_react = pd.DataFrame(abs_error_osc_react, index=basis, columns=function)
df_abs_error_osc_react.to_csv('abs_error_osc_react.csv')
df_abs_error_wavelength_prod = pd.DataFrame(abs_error_wavelength_prod, index=basis, columns=function)
df_abs_error_wavelength_prod.to_csv('abs_error_wavelength_prod.csv')
df_abs_error_wavelength_react = pd.DataFrame(abs_error_wavelength_react, index=basis, columns=function)
df_abs_error_wavelength_react.to_csv('abs_error_wavelength_react.csv')
df_abs_error_energy_prod = pd.DataFrame(abs_error_energy_prod, index=basis, columns=function)
df_abs_error_energy_prod.to_csv('abs_error_energy_prod.csv')
df_abs_error_energy_react = pd.DataFrame(abs_error_energy_react, index=basis, columns=function)
df_abs_error_energy_react.to_csv('abs_error_energy_react.csv')
df_abs_error_energy_ts = pd.DataFrame(abs_error_energy_ts, index=basis, columns=function)
df_abs_error_energy_ts.to_csv('abs_error_energy_ts.csv')
df_abs_error_tbr_energy = pd.DataFrame(abs_error_tbr_energy, index=basis, columns=function)
df_abs_error_tbr_energy.to_csv('abs_error_tbr_energy.csv')
df_abs_error_storage_energy = pd.DataFrame(abs_error_storage_energy, index=basis, columns=function)
df_abs_error_storage_energy.to_csv('abs_error_storage_energy.csv')