import pyxrd as px
from PIL import Image
import numpy as np

# Load the diffraction pattern from a JPEG file
image = Image.open('diffraction_pattern.jpg')  # Replace with your diffraction pattern JPEG file
diffraction_pattern = np.array(image)

# Create a DiffractionPattern object
dp = px.DiffractionPattern(diffraction_pattern)

# Perform indexing
results = dp.indexing()

# Print the indexed results
for solution in results:
    print("Solution:")
    print("    Unit cell: ", solution.unit_cell)
    print("    Orientation: ", solution.orientation)
    print("    Confidence: ", solution.confidence)
    print()
