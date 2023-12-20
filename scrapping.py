import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_dft = pd.read_pickle('/groups/kemi/obel/azobenzene/compchem/comparison/dft_results.pkl')


true_osc_prod = 0
true_osc_reac = 0
true_wavelength_prod = 0
true_wavelength_reac = 0
true_energy_prod = 0
true_energy_reac = 0
true_energy_ts = 0
true_tbr_energy = 0
true_storage_energy = 0
true_sol_conv_eff = 0

basissets = df_dft['basis'].unique()
basis = ['STO-3G', '6-31++Gdp', '6-311++Gdp', 'pc-0', 'pc-1', 'pc-2', 'aug-pc-0', 'aug-pc-1']
functionals = df_dft['function'].unique()
function = ['LSD','PBE0','B3LYP','B3PW91','CAM-B3LYP','wB97X-D3','M062X','CAM-B3LYP-D4','B2PLYP']
            
abs_error_osc_prod = np.zeros((len(basis), len(function)))
abs_error_osc_reac = np.zeros((len(basis), len(function)))
abs_error_wavelength_prod = np.zeros((len(basis), len(function)))
abs_error_wavelength_reac = np.zeros((len(basis), len(function)))
abs_error_energy_prod = np.zeros((len(basis), len(function)))
abs_error_energy_reac = np.zeros((len(basis), len(function)))
abs_error_energy_ts = np.zeros((len(basis), len(function)))
abs_error_tbr_energy = np.zeros((len(basis), len(function)))
abs_error_storage_energy = np.zeros((len(basis), len(function)))
abs_error_sol_conv_eff = np.zeros((len(basis), len(function)))

for i in range(len(basis)):
    for j in range(len(function)):
        print(basis[i], function[j])
        df_temp = df_dft[(df_dft['basis'] == basis[i]) & (df_dft['function'] == function[j])]
        
        # Assign the value for each property with error catch
        if len(df_temp['osc_prod'][0]) == 0:
            abs_error_osc_prod[i][j] = -1
        else:
            abs_error_osc_prod[i][j] = df_temp['osc_prod'][0][0] - true_osc_prod
        if len(df_temp['osc_reac'][0]) == 0:
            abs_error_osc_reac[i][j] = -1
        else:
            abs_error_osc_reac[i][j] = df_temp['osc_reac'][0][0] - true_osc_reac
        if len(df_temp['wavelength_prod'][0]) == 0:
            abs_error_wavelength_prod[i][j] = -1
        else:
            abs_error_wavelength_prod[i][j] = df_temp['wavelength_prod'][0][0] - true_wavelength_prod
        if len(df_temp['wavelength_reac'][0]) == 0:
            abs_error_wavelength_reac[i][j] = -1
        else:
            abs_error_wavelength_reac[i][j] = df_temp['wavelength_reac'][0][0] - true_wavelength_reac


        if len(df_temp['energy_prod']) == 0:
            abs_error_energy_prod[i][j] = -1
        else:   
            abs_error_energy_prod[i][j] = df_temp['energy_prod'] - true_energy_prod
        if len(df_temp['energy_reac']) == 0:  
            abs_error_energy_reac[i][j] = -1   
        else:
            abs_error_energy_reac[i][j] = df_temp['energy_reac'] - true_energy_reac
        if len(df_temp['energy_ts']) == 0:
            abs_error_energy_ts[i][j] = -1
        else:
            abs_error_energy_ts[i][j] = df_temp['energy_ts'] - true_energy_ts
        if len(df_temp['tbr_energy']) == 0:
            abs_error_tbr_energy[i][j] = -1
        else:
            abs_error_tbr_energy[i][j] = df_temp['tbr_energy'] - true_tbr_energy

        if len(df_temp['storage_energy']) == 0:
            abs_error_storage_energy[i][j] = -1
        else:
            abs_error_storage_energy[i][j] = df_temp['storage_energy'] - true_storage_energy

        if len(df_temp['solar_conversion_efficiency']) == 0:
            abs_error_sol_conv_eff[i][j] = -1
        else:
            abs_error_sol_conv_eff[i][j] = df_temp['solar_conversion_efficiency'] - true_sol_conv_eff

        # abs_error_energy_prod[i][j] = np.abs(df_temp['energy_prod'] - true_energy_prod)
        # abs_error_energy_reac[i][j] = np.abs(df_temp['energy_reac'] - true_energy_reac)
        # abs_error_energy_ts[i][j] = np.abs(df_temp['energy_ts'] - true_energy_ts)
        # abs_error_tbr_energy[i][j] = np.abs(df_temp['tbr_energy'] - true_tbr_energy)
        # abs_error_storage_energy[i][j] = np.abs(df_temp['storage_energy'] - true_storage_energy)


# All examined properties
properties = ['osc_prod', 'osc_reac', 'wavelength_prod', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts', 'tbr_energy', 'storage_energy','sol_conv_eff']
relevant_properties = ['wavelength_prod', 'wavelength_reac', 'storage_energy', 'tbr_energy', 'solar_conversion_efficiency','osc_prod', 'osc_reac']


# check for outliers


limit = 400*2625.5


#convert energy from hartree to kJ/mol
abs_error_energy_prod = abs_error_energy_prod*2625.5
abs_error_energy_reac = abs_error_energy_reac*2625.5
abs_error_energy_ts = abs_error_energy_ts*2625.5
abs_error_tbr_energy = abs_error_tbr_energy*2625.5
abs_error_storage_energy = abs_error_storage_energy*2625.5


for i in properties:
    abs_error = eval('abs_error_'+i)
    if max(abs_error.flatten()) > limit:
        print(i, max(abs_error.flatten()))
        print(np.where(abs_error == max(abs_error.flatten())))
        #set to -1
        abs_error[np.where(abs_error >= limit)] = -1

# Plot the absolute error for each property as a seperate heatmap  
# Handle the error catch for the properties that are not NaN
for i in properties:
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(np.abs(eval('abs_error_'+i)), annot=True, ax=ax, xticklabels=function, yticklabels=basis, cmap='viridis', fmt='.2f')
    ax.set_title('Absolute Error for '+i)
    ax.set_xlabel('Functional')
    ax.set_ylabel('Basis')
    plt.savefig('dfb_values_'+i+'.png', dpi=300)
    plt.close()