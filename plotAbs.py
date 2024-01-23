import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
import os
import sys
import glob

from calc_sce import *


colors = [
    "#2380a8",  # Dark Sky Blue
    "#FF7F50",  # Coral
    "#008080",  # Teal
    "#E6E6FA",  # Lavender
    "#808000",  # Olive Green
    "#FFA500",  # Saffron
    "#2F4F4F",  # Dark Slate Gray
    "#9966CC",  # Amethyst
    "#98FF98",  # Mint Green
    "#DE3163",  # Cerise
    "#4B0082"   # Indigo
]

line_styles = [
    "-",  # solid line
    "--", # dashed line
    "-.", # dash-dot line
    ":",  # dotted line
    (0, (3, 10, 1, 10)),  # dash-dot pattern with custom spacing
    (0, (3, 5, 1, 5)),    # another custom dash-dot pattern
    (0, (5, 10)),         # long dashes
    (0, (1, 10)),         # very short dashes
    (0, (3, 1, 1, 1)),    # dash-dot-dot pattern
    (0, (5, 1)),          # long dash, short space
    (0, (3, 5, 1, 5, 1, 5)) # custom complex pattern
]




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


# plt.savefig('casscf_abs.pdf')

def plot_abs():
    # # Plot reactant absorption spectra

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
    plt.savefig('reactant_abs.pdf')

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
    plt.savefig('product_abs.pdf')


def plot_abs_prod():
    # Plot Wavelength vs. Oscillator strength as vertical lines
    fig, ax = plt.subplots(figsize=(16,16))
    plt.title('cis-Azobenzene absorption spectra')
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
    plt.savefig('osc_prod.pdf')

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
    plt.savefig('osc_reac.pdf')

def reference_abs():
    df = abs_df[abs_df['function'] == 'casscf']
    df = df[df['basis'] == 'aug-cc-pVTZ']
    return df['broadened_intensities_reactant'].values[0],df['broadened_intensities_product'].values[0]

def plot_abs_by_functional(basis):
    # Plot reactant absorption spectra
    fig, ax = plt.subplots(figsize=(12,12),dpi=300)
    plt.title('trans-Azobenzene absorption spectra')
    plt.xlabel('Wave length (nm)')
    plt.ylabel('Intensity')

    reactant_ref, product_ref = reference_abs()
    print(max(reactant_ref))
    print(max(product_ref))
    energy_values_ref = 1239.8/abs_df['energy_values'][0]
    ref_prod_max_idx = np.argmax(product_ref)
    ref_reac_max_idx = np.argmax(reactant_ref)
    print(energy_values_ref[ref_prod_max_idx])
    print(energy_values_ref[ref_reac_max_idx])    


    sns.lineplot(x=energy_values_ref,y=reactant_ref,label=f'CASSCF',color=colors[0],linestyle=line_styles[1])
    color_idx = 1
    for i in range(len(abs_df)):
        # print(i)
        function = abs_df['function'][i]
        if function == 'B2PLYP': continue
        if basis != abs_df['basis'][i]: continue
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
        sns.lineplot(x=energy_values,y=broadened_intensities_reactant,label=f'{function}, {basis}',color=colors[color_idx],linestyle=line_styles[0])
        color_idx += 1
    # remove legend
    
    plt.xlim(270,900)
    plt.savefig(f'abs_plot/basis_reac/reactant_abs_{basis}.pdf')

    # Plot product absorption spectra
    fig, ax = plt.subplots(figsize=(12,12),dpi=300)
    plt.title('cis-Azobenzene absorption spectra')
    plt.xlabel('Wave length (nm)')
    plt.ylabel('Intensity')

    sns.lineplot(x=energy_values_ref,y=product_ref,label=f'CASSCF',color=colors[0],linestyle=line_styles[1])
    
    color_idx = 1
    for i in range(len(abs_df)):
        # print(i)
        function = abs_df['function'][i]
        if function == 'B2PLYP': continue
        if basis != abs_df['basis'][i]: continue
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
        sns.lineplot(x=energy_values,y=broadened_intensities_product,label=f'{function}, {basis}',color=colors[color_idx],linestyle=line_styles[0])
        color_idx += 1

    plt.xlim(270,900)
    plt.savefig(f'abs_plot/basis_prod/product_abs_{basis}.pdf')


def plot_abs_by_basis_set(functional):
    # Plot reactant absorption spectra
    fig, ax = plt.subplots(figsize=(12,12),dpi=300)
    plt.title('trans-Azobenzene Absorption spectra')
    plt.xlabel('Wave length (nm)')
    plt.ylabel('Intensity')

    reactant_ref, product_ref = reference_abs()
    energy_values = abs_df['energy_values'][0]
    sns.lineplot(x=1239.8/energy_values,y=reactant_ref,label=f'CASSCF',color=colors[0],linestyle=line_styles[1])

    color_idx = 1
    for i in range(len(abs_df)):
        # print(i)
        basis = abs_df['basis'][i]
        if functional != abs_df['function'][i]: continue
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
        sns.lineplot(x=energy_values,y=broadened_intensities_reactant,label=f'{functional}, {basis}',color=colors[color_idx],linestyle=line_styles[0])
        color_idx += 1
    # remove legend
    
    plt.xlim(270,900)
    plt.savefig(f'abs_plot/func_reac/reactant_abs_{functional}.pdf')

    # Plot product absorption spectra
    fig, ax = plt.subplots(figsize=(12,12),dpi=300)
    plt.title('cis-Azobenzene Absorption spectra')
    plt.xlabel('Wave length (nm)')
    plt.ylabel('Intensity')

    sns.lineplot(x=energy_values,y=product_ref,label=f'CASSCF',color=colors[0],linestyle=line_styles[1])
    
    color_idx = 1
    for i in range(len(abs_df)):
        # print(i)
        basis = abs_df['basis'][i]
        if functional != abs_df['function'][i]: continue
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
        sns.lineplot(x=energy_values,y=broadened_intensities_product,label=f'{functional}, {basis}',color=colors[color_idx],linestyle=line_styles[0])
        color_idx += 1

    plt.xlim(270,900)
    plt.savefig(f'abs_plot/func_prod/product_abs_{functional}.pdf')







for functional in abs_df['function'].unique():
    if functional == 'B2PLYP': continue
    if functional == 'casscf': continue
    plot_abs_by_basis_set(functional)

for basis in abs_df['basis'].unique():
    if basis == 'aug-cc-pVTZ': continue
    plot_abs_by_functional(basis)
