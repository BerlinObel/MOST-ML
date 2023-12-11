# Import relevant modules
import numpy as np
import pandas as pd
import os
import sys
import glob



verbose = True
# File paths
# DFT
dft_path_es = '/groups/kemi/obel/azobenzene/compchem/benchmark/esDynamics/done/'
dft_path = '/groups/kemi/obel/azobenzene/compchem/done/'

# Initial files to collect
react_files = glob.glob(dft_path + 'react/azo_r_*.out')


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
    # reac = glob.glob(dft_path + 'react/azo_r_*'+basis+'*'+function+'*.out')
    prod = glob.glob(dft_path + 'prod/azo_p_*'+basis+'*'+function+'*.out')
    ts = glob.glob(dft_path + 'ts/azo_ts_*'+basis+'*'+function+'*.out')

    # Return the newest files using glob if multiple files are found
    if len(prod) > 1:
        prod = max(prod, key=os.path.getctime)
    if len(ts) > 1:
        ts = max(ts, key=os.path.getctime)
    
    return es_reac[0], es_prod[0], prod, ts


# Define function to collect DFT Energies
def collectDFT(file):
    # Read in DFT data
    lines = open(file).readlines()
    for line in lines:
        if 'Total Enthalpy' in line:
            energy = float(line.split()[2])

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
# df columns: file, function, basis, osc_prod, wavelength_prod, osc_react, wavelength_react, energy_prod, energy_react, energy_ts, tbr_energy, storage_energy 
def collectAll():
    # Initialize dataframe
    df = pd.DataFrame(columns=['file', 'function', 'basis', 'osc_prod', 'wavelength_prod', 'osc_react', 'wavelength_react', 'energy_prod', 'energy_react', 'energy_ts'])
    # Collect files
    for file in react_files:
        function, basis = read_functional_and_basis(file)
        if verbose: print('Collecting files for functional: {} and basis set: {}'.format(function, basis))
        es_reac, es_prod, prod, ts = get_files(function, basis)
        if verbose: print('Collecting data from files: {} {} {} {}'.format(es_reac, es_prod, prod, ts))
        osc_prod, wavelength_prod = collectES(es_prod)
        if verbose: print('Oscillators and wavelengths for product: {} {}'.format(osc_prod, wavelength_prod))
        osc_reac, wavelength_reac = collectES(es_reac)
        if verbose: print('Oscillators and wavelengths for reactant: {} {}'.format(osc_reac, wavelength_reac))
        energy_prod = collectDFT(prod)
        if verbose: print('Energy of product: {}'.format(energy_prod))
        energy_reac = collectDFT(file)
        if verbose: print('Energy of reactant: {}'.format(energy_reac))
        energy_ts = collectDFT(ts)
        if verbose: print('Energy of transition state: {}'.format(energy_ts))
        tbr_energy = energy_ts - energy_prod
        if verbose: print('TBR Energy: {}'.format(tbr_energy))
        storage_energy = energy_prod - energy_reac
        if verbose: print('Storage Energy: {}'.format(storage_energy))
        # Append to dataframe
        df = df.append({'file': file, 'function': function, 'basis': basis, 'osc_prod': osc_prod, 'wavelength_prod': wavelength_prod, 'osc_react': osc_reac, 'wavelength_react': wavelength_reac, 'energy_prod': energy_prod, 'energy_react': energy_reac, 'energy_ts': energy_ts, 'tbr_energy': tbr_energy, 'storage_energy': storage_energy}, ignore_index=True)
    return df

# Run function
df = collectAll()

# Save dataframe
df.to_pickle('dft_results.pkl')