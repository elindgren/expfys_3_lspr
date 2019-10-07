import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from data_loader import load_file, read_gas_file
from fit_function import lorentzian_fit
import seaborn as sns
sns.set_palette(sns.color_palette("husl", 20))


# Set plot params
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

# Load the data
filename = 'Ag_diskrod_200nm_15pt'
lamp_spectrum_file = 'CRS_700nm'
background_spectra = 15

# Size of sensor is 1024x256
data, nbr_particles = load_file(filename)
lamp_data, s = load_file(lamp_spectrum_file)

# Plot the data
fig, ax = plt.subplots(figsize=(10,6))
ax = fig.gca(projection='3d')
wvl = data["lambda"]
background = data["spectra_" + str(background_spectra)]
lamp_spectra = lamp_data["spectra_0"]

#peak_guesses = [0, 0, 750, 700, 650, 600, 550, 500, 450, 400, 300, 350, 250]
peak_positions = []
for i in range(0,nbr_particles):
    spectra = data[f'spectra_{i}']
    if not i == background_spectra:
        # Don't plot background
        corrected_spectra = (spectra - background)/lamp_spectra
        #norm_spectra = spectra/spectra.max()

        peak_pos, fwhm = lorentzian_fit(wvl, corrected_spectra, [700, 850], False)
        peak_positions.append(peak_pos)
        delta_y = np.ones(len(wvl))*(i)
        ax.plot(wvl, delta_y, corrected_spectra, label=f'Particle {i+1}', alpha=0.7)

# Plot peak position as well
ax.plot(peak_positions, list(range(nbr_particles-1)), np.zeros(nbr_particles-1), 'k--', label="Peak position")

#ax.set_title(f'Corrected spectra for file {filename}')
ax.tick_params(axis='both', which='major', pad=0)
ax.set_xlabel("Wavelength")
ax.set_ylabel("Measurement")
ax.set_zlabel("Counts")
plt.grid()
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f'{filename}.png')
plt.show()

# Centroid = center of mass wavelength
# Normalize spectra for a better fit - best around 0 to 1
