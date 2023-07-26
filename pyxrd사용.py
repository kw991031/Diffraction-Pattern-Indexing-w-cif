import tkinter as tk
from tkinter import filedialog
import pyxrd

# Create a Tkinter window
root = tk.Tk()
root.withdraw()

# Open a file dialog to choose the CIF file
file_path = filedialog.askopenfilename(title='Select CIF File', filetypes=[('CIF Files', '*.cif')])
if not file_path:
    print("No file selected.")
    exit()

# Load the crystal structure from the CIF file
crystal = pyxrd.load_crystal_from_cif(file_path)

# Set the scattering factors for the elements in the crystal
scattering_factors = {
    'Li': 3.007,
    'Mn': 5.557,
    'Ni': 6.644,
    'Mg': 3.368,
    'O': 3.465,
}

# Set the X-ray wavelength in Angstrom
wavelength = 1.5418

# Calculate the structure factor
structure_factor = pyxrd.calc_structure_factor(crystal, scattering_factors, wavelength)

# Print the structure factor
print(structure_factor)
