# Import relevant modules
import numpy as np
import pandas as pd
import os
import sys
import glob

from calc_sce import *

xtb_compare = False
verbose = False
# File paths
# DFT
dft_path_es = '/groups/kemi/obel/azobenzene/compchem/benchmark/esDynamics/done/'
dft_path = '/groups/kemi/obel/azobenzene/compchem/done/'

if xtb_compare:
    dft_path_es = '/groups/kemi/obel/dft_calc/esDynamics/'
    dft_path = '/groups/kemi/obel/dft_calc/done/outfiles/'

# Initial files to collect
reac_files = glob.glob(dft_path + 'azo_r_*.out')


# Define function to read functional and basis set from file name
def read_functional_and_basis(file):
    file = file.split('_')
    functional = file[3].split('.')[0]
    basis = file[2]
    return functional, basis

# Get all the files for a given basis set and functional
def get_files(function, basis):

    es_reac = glob.glob(dft_path_es + 'azo_r_'+basis+'*'+function+'*.out')
    es_prod = glob.glob(dft_path_es + 'azo_p_'+basis+'*'+function+'*.out')
    # reac = glob.glob(dft_path + 'reac/azo_r_*'+basis+'*'+function+'*.out')
    prod = glob.glob(dft_path + 'prod/azo_p_*'+basis+'*'+function+'*.out')
    ts = glob.glob(dft_path + 'ts/azo_ts_*'+basis+'*'+function+'*.out')

    # Return the newest files using glob if multiple files are found
    if len(prod) > 1:
        prod = [max(prod, key=os.path.getctime)]
    if len(ts) > 1:
        ts = [max(ts, key=os.path.getctime)]
    
    if not es_reac:
        print(es_reac)
        es_reac = ['None']
    if not es_prod: 
        es_prod = ['None']
    if not ts:
        print(ts)
        ts = ['None']
    if not prod: 
        prod = ['None']
    
    return es_reac[0], es_prod[0], prod[0], ts[0]


# Define function to collect DFT Energies
def collectDFT(file):
    # Read in DFT data
    lines = open(file).readlines()
    for line in lines:
        if 'Total Enthalpy' in line:
            energy = float(line.split()[3])

    # Return Energy
    return energy

# Define function to collect DFT Absolute Energies
def collectES(file):
    # Read in DFT data
    lines = open(file).readlines()
    es_lines = []
    for i, line in enumerate(lines):
        if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
            es_lines = lines[i+5:i+10]
    osc = []
    wavelength = []
    for line in es_lines:
        osc.append(float(line.split()[3]))
        wavelength.append(float(line.split()[2]))
    
    return osc, wavelength



# Define function to collect all results in a dataframe
# df columns: file, function, basis, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, solar_conversion_efficiency
def collectAll():
    # Initialize dataframe
    df = pd.DataFrame(columns=['file', 'function', 'basis', 'osc_prod', 'wavelength_prod', 'osc_reac', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts', 'tbr_energy', 'storage_energy', 'solar_conversion_efficiency'])
    # Collect files
    print('Collecting files' )
    for file in reac_files:
        print(file)
        function, basis = read_functional_and_basis(file)

        if basis == 'pc-3' or basis == 'pc-4' or basis == 'aug-pc-3' or basis == 'aug-pc-4' or basis == 'aug-pc-2': continue
        if verbose: print('Collecting files for functional: {} and basis set: {}'.format(function, basis))
        es_reac, es_prod, prod, ts = get_files(function, basis)
        if verbose: print('Collecting data from files: {} {} {} {}'.format(es_reac, es_prod, prod, ts))
        if es_reac == 'None' or es_prod == 'None':
            osc_prod, wavelength_prod = [],[]
            osc_reac, wavelength_reac = [],[]
            print('No ES files found for functional: {} and basis set: {}'.format(function, basis))
        else:     
            osc_prod, wavelength_prod = collectES(es_prod)
            if verbose: print('Oscillators and wavelengths for product: {} {}'.format(osc_prod, wavelength_prod))
            osc_reac, wavelength_reac = collectES(es_reac)
            if verbose: print('Oscillators and wavelengths for reacant: {} {}'.format(osc_reac, wavelength_reac))
        energy_prod = collectDFT(prod)
        if verbose: print('Energy of product: {}'.format(energy_prod))
        energy_reac = collectDFT(file)
        if verbose: print('Energy of reacant: {}'.format(energy_reac))
        if ts == 'None':
            energy_ts = -1
            print('No TS file found for functional: {} and basis set: {}'.format(function, basis))
        else: energy_ts = collectDFT(ts)
        if verbose: print('Energy of transition state: {}'.format(energy_ts))
        tbr_energy = energy_ts - energy_prod
        if verbose: print('TBR Energy: {}'.format(tbr_energy))
        storage_energy = energy_prod - energy_reac
        if verbose: print('Storage Energy: {}'.format(storage_energy))
        if es_reac == 'None' or es_prod == 'None':
            sce = -1
            # print('No ES files found for functional: {} and basis set: {}'.format(function, basis))
        elif energy_ts == -1:
            sce = -1
            # print('No TS file found for functional: {} and basis set: {}'.format(function, basis))
        if function == 'B2PLYP' : sce = 0
        else: sce = calculate_SCE(storage_energy,tbr_energy,wavelength_reac[0],wavelength_prod[0],osc_reac[0],osc_prod[0])
        if sce < 1: print('Solar Conversion Efficiency: {}, Storage Energy: {}, TBR Energy: {}, Wavelength: {}, Oscillator: {}'.format(sce, storage_energy, tbr_energy, wavelength_prod[0], osc_prod[0]))
        if verbose: print('Solar Conversion Efficiency: {}'.format(sce))

        # Append to dataframe using concat
        df = pd.concat([df, pd.DataFrame([[file, function, basis, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, sce]], columns=['file', 'function', 'basis', 'osc_prod', 'wavelength_prod', 'osc_reac', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts', 'tbr_energy', 'storage_energy', 'solar_conversion_efficiency'])])
    return df

# Run function
df = collectAll()


# Save dataframe
df.to_pickle('dft_results.pkl')

print('Done!')
print('Dataframe saved to dft_results.pkl')
print('Dataframe shape: {}'.format(df.shape))
print('Dataframe columns: {}'.format(df.columns))
print('Dataframe head: {}'.format(df.head()))
