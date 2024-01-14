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
path = '/groups/kemi/obel/azobenzene/casscf/'
path_p_es = path + 'azo_p2_casscf_singlet.out'
path_p = path + 'azo_p_casscf.out'
path_r_es = path + 'azo_r2_casscf_singlet.out'
path_r = path + 'azo_r2_casscf.out'
path_ts = path + 'azo_ts_casscf.out'

# Initial dataframe
# df columns: file, function, basis, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts, tbr_energy, storage_energy, solar_conversion_efficiency
df = pd.DataFrame(columns=['file', 'function', 'basis', 'osc_prod', 'wavelength_prod', 'osc_reac', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts', 'tbr_energy', 'storage_energy', 'solar_conversion_efficiency'])

def collectES(file):
    # Read in DFT data
    osc = []
    wavelength = []
    lines = open(file).readlines()
    for i, line in enumerate(lines):
        if '    ABSORPTION SPECTRUM' in line :
            # Get the oscillator strength and wavelength
            es_lines = lines[i+5:i+9]
    for line in es_lines:
        line = line.split()
        osc.append(float(line[7]))
        wavelength.append(float(line[6]))
    return osc, wavelength

def collectCASSCF(file):
    # Read in DFT data and reverse the list
    lines = open(file).readlines()
    lines.reverse()

    for line in lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            energy = float(line.split()[4])
    return energy

file = 'azo_casscf'
function = 'casscf'
basis = 'aug-cc-pVTZ'
osc_prod, wavelength_prod = collectES(path_p_es)
osc_reac, wavelength_reac = collectES(path_r_es)
energy_prod = collectCASSCF(path_p)
energy_reac = collectCASSCF(path_r)
energy_ts = collectCASSCF(path_ts)
tbr_energy = energy_ts - energy_prod
storage_energy = energy_prod - energy_reac
sce = calculate_SCE(storage_energy,tbr_energy,wavelength_reac[0],wavelength_prod[0],osc_reac[0],osc_prod[0])
# Print all nice and pretty
print('File: {}'.format(file))
print('Function: {}'.format(function))
print('Basis: {}'.format(basis))
print('Oscillator Strengths (Prod): {}'.format(osc_prod))
print('Wavelengths (Prod): {}'.format(wavelength_prod))
print('Oscillator Strengths (reac): {}'.format(osc_reac))
print('Wavelengths (reac): {}'.format(wavelength_reac))
print('Energy (Prod): {}'.format(energy_prod))
print('Energy (reac): {}'.format(energy_reac))
print('Energy (TS): {}'.format(energy_ts))
print('TBR Energy: {}'.format(tbr_energy * 2625.5))
print('Storage Energy: {}'.format(storage_energy * 2625.5))
print('Solar Conversion Efficiency: {}'.format(sce))

df = pd.concat([df, pd.DataFrame([[file, function, basis, osc_prod, wavelength_prod, osc_reac, wavelength_reac, energy_prod, energy_reac, energy_ts,tbr_energy,storage_energy,sce]], columns=['file', 'function', 'basis', 'osc_prod', 'wavelength_prod', 'osc_reac', 'wavelength_reac', 'energy_prod', 'energy_reac', 'energy_ts','tbr_energy','storage_energy','solar_conversion_efficiency'])])

df.to_pickle('casscf_results.pkl')