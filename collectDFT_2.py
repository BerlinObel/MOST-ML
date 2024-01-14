# Import relevant modules
import numpy as np
import pandas as pd
import os
import sys
import glob

from calc_sce import *


verbose = True
# File paths
# DFT

dft_path_es = '/groups/kemi/obel/dft_calc/esDynamics/'
dft_path = '/groups/kemi/obel/dft_calc/done/outfiles/'

# Initial files to collect
reac_files = glob.glob(dft_path + '*_r_*.out')

if verbose:
    print(reac_files)

def read_hash(file):
    file = file.split('_')
    hash = file[2]
    return hash

# Get all the files for a given basis set and functional
def get_files(hash):
    es_reac = glob.glob(dft_path_es + 'azo_'+hash+'_r_'+'*.out')
    es_prod = glob.glob(dft_path_es + 'azo_'+hash+'_p_'+'*.out')
    prod = glob.glob(dft_path + 'azo_'+hash+'_p_'+'*.out')
    
    if not es_reac:
        es_reac = ['None']
    if not es_prod: 
        es_prod = ['None']
    if not prod: 
        prod = ['None']
    
    return es_reac[0], es_prod[0], prod[0]


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
# df columns: file, hash, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, solar_conversion_efficiency
def collectAll():
    # Initialize dataframe
    df = pd.DataFrame(columns=['file', 'hash', 'osc_prod', 'wavelength_prod', 'osc_reac', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts', 'tbr_energy', 'storage_energy', 'solar_conversion_efficiency'])
    # Collect files
    for file in reac_files:
        hash = read_hash(file)
        if verbose:
            print(hash)
        es_reac, es_prod, prod = get_files(hash)
        if verbose:
            print(es_reac, es_prod, prod)
        # Read excited state data
        if es_reac != 'None':
            osc_reac, wavelength_reac = collectES(es_reac)
        else:
            osc_reac = ['None']
            wavelength_reac = ['None']
        if es_prod != 'None':
            osc_prod, wavelength_prod = collectES(es_prod)
        else:        
            osc_prod = ['None']
            wavelength_prod = ['None']

        
        # Read DFT data
        energy_reac = collectDFT(file)

        if prod != 'None':
            energy_prod = collectDFT(prod)
        else:   
            energy_prod = energy_reac
        if verbose:
            print(energy_prod, energy_reac)

        # Transition state data was not collected so set to 120 to allow for calculation of SCE
        energy_ts = 120

        # Calculate SCE
        tbr_energy = energy_prod - energy_reac
        storage_energy = energy_prod - energy_ts
        if verbose:
            print(tbr_energy, storage_energy)

        # Calculate solar conversion efficiency
        sce = calculate_SCE(storage_energy,tbr_energy,wavelength_reac[0],wavelength_prod[0],osc_reac[0],osc_prod[0])

        # Append to dataframe using concat
        df = pd.concat([df, pd.DataFrame([[file, hash, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, sce]], columns=['file', 'hash', 'osc_prod', 'wavelength_prod', 'osc_reac', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts', 'tbr_energy', 'storage_energy', 'solar_conversion_efficiency'])])
    return df

# Run function
df = collectAll()

# Save dataframe
df.to_pickle('dft_33_systems.pkl')