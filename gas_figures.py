import numpy as np
import matplotlib.pyplot as plt
from data_loader import read_gas_file
from data_loader import load_file
from data_loader import read_timeseries
from fit_function import lorentzian_fit
import pandas as pd
from tqdm import tqdm
import tikzplotlib
import seaborn as sns
sns.set_palette(sns.husl_palette(20, h=.7))

# Set plot params
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

# Files
filename = 'Au_diskrod_200nm_15pt_pulse'
lamp_file = 'CRS_700nm_glas'
gas_file = 'Lab_TIF295_ArH2_pulses_Au'
nbr_particles = 15
samples_to_plot = [11]

# Load lamp spectra
lamp_data, _  = load_file(lamp_file)
lamp_spectra = lamp_data["spectra_0"]

# Load gas data - ndarray [t,h2]
gas_data = read_gas_file(gas_file)
#fig_gas, ax_gas = plt.subplots()
#ax_gas.plot(gas_data[0], gas_data[1])

# Load data
measurements = read_timeseries(filename, nbr_particles) 
#print(measurements)

peak_guess = 750  # About 750 nm is a good guess for Pd

# Loop over all measurements at different times t
peak_positions = [[] for i in range(nbr_particles)]
for iter, measurement_df in tqdm(enumerate(measurements)):
    wvl = measurement_df.iloc[:,0]
    b_ground = measurement_df.iloc[:,1]
    # Find peak position and FWHM for each particle
    for measurement_nbr in measurement_df:
        if measurement_nbr in samples_to_plot:
            # The first two columns are always wavelength and background
            spectra = measurement_df.iloc[:, measurement_nbr]
            corr_spectra =  (spectra-b_ground)/lamp_spectra
            peak_pos, fwhm = lorentzian_fit(wvl, corr_spectra, peak_guess, False)
            peak_positions[measurement_nbr].append(fwhm)

# Take only every 30:th point from the gas_data measurements
t = np.array([a for idx, a in enumerate(gas_data[0]) if idx%30 == 0])
g = np.array([a for idx, a in enumerate(gas_data[1]) if idx%30 == 0])
# Remove the extra measurements from gas measurement
t = t[:len(measurements)]
g = g[:len(measurements)]
# Plot
fig, ax1 = plt.subplots(figsize=(10,3))
#for particle_peak_pos in peak_positions:
#t =  np.arange(0,119)*30
lns = []
for smpl in samples_to_plot:
    # Plot delta FWHM relative to smallest FWHM for each sample (remove offset)
    sample_series = np.array(peak_positions[smpl])
    smallest_fwhm = np.amin(sample_series)
    ln = ax1.plot(t, sample_series-smallest_fwhm, linewidth=2, color='C10', label=f'Au')
    lns += ln

ax2 = ax1.twinx()
ln = ax2.plot(t, g/2, 'g--', label=r'H$_2$')
lns += ln
ax1.grid(True)
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=2)

ax2.set_ylim(0,30)
plt.locator_params(axis='y', nbins=6)
ax1.set_xlabel(r'Time [s]')
ax1.set_ylabel(r'$\Delta$FWHM [nm]')
ax2.set_ylabel(r'H$_2$ [% vol]')

plt.tight_layout()
plt.savefig(f'{filename}.png')
tikzplotlib.save(f'{filename}.tex')
plt.show()


