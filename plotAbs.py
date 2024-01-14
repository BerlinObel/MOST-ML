import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
import os
import sys
import glob

from calc_sce import *

try:
    am15_sheet = pd.read_pickle('am15_SMARTS2.pkl')
except:
    am15_path = "/lustre/hpc/kemi/elholm/bod_calc/alle/eff/am_15g.xls"
    am15_sheet = pd.read_excel(am15_path,sheet_name = "SMARTS2",header=None)
    pd.to_pickle(am15_sheet,'am15_SMARTS2.pkl')


# Read solar spectrum - chosen spectrum normalized to 1000 (1000 W/m2)
solar_spectrum = am15_sheet.to_numpy()
wavelengths = solar_spectrum[2:, 0]
solar_irradiance = solar_spectrum[2:, 2]



# # load data
# df_files = glob.glob('*.pkl')
# dfs = {}
# for file in df_files:
#     if 'am15' in file: continue
#     if 'abs' in file: continue
#     if '33' in file: continue
#     dfs[file] = pd.read_pickle(file)

# # plot data
# # df columns: file, function, basis, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, solar_conversion_efficiency
# # df columns: file, function, basis, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, solar_conversion_efficiency

# abs_df = pd.DataFrame(columns=['file','function','basis','energy_values','broadened_intensities_reactant','broadened_intensities_product'])
# def read_abs_data():
#     for file in dfs:
#         print(file)
#         df = dfs[file]
#         df = df.reset_index(drop=True)
#         for i in range(len(df)):
#             print(i) 
#             try:
#                 function = df['function'][i]
#                 basis = df['basis'][i]
#             except:
#                 function = 'CASSCF'
#                 basis = 'aug-cc-pVTZ'
            
#             if function == 'B2PLYP': continue
#             osc_prod = df['osc_prod'][i][0]
#             wavelength_prod = df['wavelength_prod'][i][0]
#             osc_reac = df['osc_reac'][i][0]
#             wavelength_reac = df['wavelength_reac'][i][0]


#             print(f'Function: {function}, Basis: {basis}, Osc_prod: {osc_prod}, Wavelength_prod: {wavelength_prod}, Osc_reac: {osc_reac}, Wavelength_reac: {wavelength_reac}')
#             excitation_energy_eV = 1239.8 / wavelength_reac
#             energy_values, broadening_intensities_reactant = apply_gaussian_broadening(excitation_energy_eV, osc_reac,wavelengths,0.4)
#             excitation_energy_eV = 1239.8 / wavelength_prod
#             energy_values, broadening_intensities_product = apply_gaussian_broadening(excitation_energy_eV, osc_prod,wavelengths,0.4)

#             abs_df = abs_df._append({'file':file,'function':function,'basis':basis,'energy_values':energy_values,'broadened_intensities_reactant':broadening_intensities_reactant,'broadened_intensities_product':broadening_intensities_product},ignore_index=True)


#     abs_df.to_pickle('abs_df.pkl')

# read_abs_data()
# 
abs_df = pd.read_pickle('abs_df.pkl')
# # plot data
# print(abs_df)
# # fig, ax = plt.subplots(figsize=(16,16))
# # plt.plot([1, 2, 3, 4])
# # plt.ylabel('some numbers')
# # plt.show()

# #plot casscf data
# casscfdf = abs_df[abs_df['file'] == 'casscf_results.pkl']
# print(casscfdf)
# cas_energy_values = casscfdf['energy_values'].values[0]
# # convert from eV to nm
# cas_energy_values = 1239.8/cas_energy_values
# cas_broadened_intensities_reactant = casscfdf['broadened_intensities_reactant'].values[0]
# cas_broadened_intensities_product = casscfdf['broadened_intensities_product'].values[0]
# print(cas_energy_values)
# print(cas_broadened_intensities_reactant)
# print(cas_broadened_intensities_product)
# sns.set_theme(style="ticks")
# fig, ax = plt.subplots(figsize=(16,16))
# plt.title('Absorption spectra')
# plt.xlabel('Energy (eV)')
# plt.ylabel('Intensity (arb. units)')
# plt.xlim(min(cas_energy_values),500)
# sns.lineplot(x=cas_energy_values,y=cas_broadened_intensities_reactant,label='Reactant')
# sns.lineplot(x=cas_energy_values,y=cas_broadened_intensities_product,label='Product')


# plt.savefig('casscf_abs.png')




# Plot reactant absorption spectra
fig, ax = plt.subplots(figsize=(16,16))
plt.title('Absorption spectra')
plt.xlabel('Wave length (nm)')
plt.ylabel('Intensity (arb. units)')

for i in range(len(abs_df)):
    print(i)
    function = abs_df['function'][i]
    basis = abs_df['basis'][i]
    energy_values = abs_df['energy_values'][i]
    broadened_intensities_reactant = abs_df['broadened_intensities_reactant'][i]
    broadened_intensities_product = abs_df['broadened_intensities_product'][i]
    # print(function)
    # print(basis)
    # print(energy_values)
    # print(broadened_intensities_reactant)
    # print(broadened_intensities_product)
    # convert from eV to nm
    energy_values = 1239.8/energy_values
    sns.lineplot(x=energy_values,y=broadened_intensities_reactant,label=f'{function}, {basis}')
# remove legend
plt.legend().remove()
plt.xlim(270,900)
plt.savefig('reactant_abs.png')

# Plot product absorption spectra
fig, ax = plt.subplots(figsize=(16,16))
plt.title('Absorption spectra')
plt.xlabel('Wave length (nm)')
plt.ylabel('Intensity (arb. units)')
for i in range(len(abs_df)):
    print(i)
    function = abs_df['function'][i]
    basis = abs_df['basis'][i]
    energy_values = abs_df['energy_values'][i]
    broadened_intensities_reactant = abs_df['broadened_intensities_reactant'][i]
    broadened_intensities_product = abs_df['broadened_intensities_product'][i]
    # print(function)
    # print(basis)
    # print(energy_values)
    # print(broadened_intensities_reactant)
    # print(broadened_intensities_product)
    # convert from eV to nm
    energy_values = 1239.8/energy_values
    sns.lineplot(x=energy_values,y=broadened_intensities_product,label=f'{function}, {basis}')
plt.legend().remove()
plt.xlim(270,900)
plt.savefig('product_abs.png')



# Plot Wavelength vs. Oscillator strength as vertical lines
fig, ax = plt.subplots(figsize=(16,16))
plt.title('Absorption spectra')
plt.xlabel('Wave length (nm)')
plt.ylabel('oscillator strength')

print('dft')
df = pd.read_pickle('dft_results.pkl')
df.sort_values(by=['function','basis'],inplace=True)
print(df['function'].unique())
print(df['basis'].unique())
df = df.reset_index(drop=True)
print(df['function'].unique())

for i in range(len(df)):
    print(i)
    function = df['function'][i]
    basis = df['basis'][i]
    print(function,basis)
    if function == 'B2PLYP': continue
    try:
        osc = df['osc_prod'][i][0]
        wavelength = df['wavelength_prod'][i][0]
    except:
        continue
    
    print(function,basis,osc,wavelength)
    
    plt.scatter(wavelength,osc,label=f'{function}, {basis}')

plt.legend()

plt.xlim(270,900)
plt.savefig('osc_prod.png')

plt.clf()
for i in range(len(df)):
    print(i)
    function = df['function'][i]
    basis = df['basis'][i]
    print(function,basis)
    if function == 'B2PLYP': continue
    try:
        osc = df['osc_reac'][i][0]
        wavelength = df['wavelength_reac'][i][0]
    except:
        continue
    
    print(function,basis,osc,wavelength)
    
    plt.scatter(wavelength,osc,label=f'{function}, {basis}')

plt.legend()

plt.xlim(270,900)
plt.savefig('osc_reac.png')


