import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from load_data import load_file, read_gas_file
import seaborn as sns
sns.set_palette(sns.color_palette("husl", 20))


# Load the data
filename = 'Group_1c/Ag_diskrod_200nm_15pt'
lamp_spectrum_file = 'Group_1c/CRS_700nm'
background_spectra = 15

# Size of sensor is 1024x256
data, nbr_particles = load_file(filename)
lamp_data, s = load_file(lamp_spectrum_file)


# Plot the data
fig, ax = plt.subplots(figsize=(8,6))
ax = fig.gca(projection='3d')
wvl = data["lambda"]
background = data["spectra_" + str(background_spectra)]
lamp_spectra = lamp_data["spectra_0"]

for i in range(0,nbr_particles):
    spectra = data[f'spectra_{i}']
    if i == background_spectra:
        c = 'k'
        a = 1
        l = "Background"
    elif i == 1 or i==nbr_particles-1:
        # First and last samples
        c = 'r'  # color
        a = 1  # alpha
        l = f'Particle {i}'  # Label
    else: 
        # All other samples
        c = 'b'
        a = 1
        l = f'Particle {i}'
    # Don't plot background
    corrected_spectra = (spectra - background)/lamp_spectra
    norm_spectra = spectra/spectra.max()
    delta_y = np.ones(len(wvl))*(i)
    ax.plot(wvl, delta_y, corrected_spectra, label=l, alpha=a)

#ax.set_title(f'Corrected spectra for file {filename}')
ax.set_xlabel("Wavelength")
ax.set_ylabel("Measurement")
ax.set_zlabel("Counts")
plt.grid()
plt.legend()

# Set font size for figure
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

plt.savefig(f'{filename}.png')
plt.show()


# Centroid = center of mass wavelength
# Normalize spectra for a better fit - best around 0 to 1
