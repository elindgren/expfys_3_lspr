import numpy as np
import pandas as pd


def load_file(filename):
    d = {}
    with open("Group_1c/" + filename + ".asc") as f:
        for idx, line in enumerate(f):
            val = line.split()
            # If first line, define dict arrays
            if idx == 0:
                nbr_samples = len(val[1:])  # The number of sampels
                d["lambda"] = []  
            # The first row is wavelength
            d["lambda"].append(float(val[0]))
            # Loop through the rest of the lines
            for i, v in enumerate(val[1:]):
                # If first line, define dict arrays
                if idx == 0:
                    d[f'spectra_{i}'] = []
                d[f'spectra_{i}'].append(float(v))
    # Convert all arrays to numpy arrays
    for key, val in d.items():
        d[key] = np.array(val)
    return d, nbr_samples


def read_gas_file(filename):
    # Load into pandas df
    df = pd.read_table(filename + '.txt', header=8)
    # Rename column labels
    df.columns = df.iloc[0]
    df = df.drop([0,1])
    # pick out relevant columns
    h2_strs = df.iloc[:, 5]
    t_strs = df.iloc[:,0]
    h2 = np.zeros(len(h2_strs))
    t = np.zeros(len(t_strs))
    # Iterate through and convert to float
    i = 0
    for h2_str, t_str in zip(h2_strs, t_strs):
        h2[i] = np.float(h2_str.replace(',', '.'))
        t[i] = np.float(t_str.replace(',', '.'))
        i += 1
    return [t, h2]

