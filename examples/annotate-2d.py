"""
Display one markers layer ontop of one image layer using the add_markers and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer
from napari.util import app_context

print("click to add markers; close the window when finished.")

with app_context():
    viewer = Viewer()
    window = Window(viewer)
    # add the image
    viewer.add_image(rgb2gray(data.astronaut()))
    # add the markers
    markers = viewer.add_markers(np.zeros((0, 2)))
    markers.mode = 'add'

print("you clicked on:")
print(markers.coords)