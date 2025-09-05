#!/usr/bin/env python
"""Visual test script for ray_tracing_resolution parameter.

This script creates a simple cube and displays it with different
ray_tracing_resolution values to visually verify the rendering quality.
"""

import numpy as np

import napari

# Create a small cube volume (5x5x5)
cube = np.ones((5, 5, 5), dtype=float)

# Create viewer in 3D mode
viewer = napari.Viewer(ndisplay=3)

# Add the cube with low quality (jagged edges expected)
layer = viewer.add_image(
    cube,
    name='cube_low_quality',
    colormap='gray',
    rendering='mip',
    ray_tracing_resolution=4.0,  # Low quality - should show jagged edges
    contrast_limits=[0, 1],
)

# Set camera for 3/4 perspective view to see edges clearly
viewer.camera.angles = (30, 45, 0)
viewer.camera.zoom = 10.0

print("Displaying cube with LOW quality (ray_tracing_resolution=4.0)")
print("You should see jagged/staircase edges on the cube")
print("Press 'c' to change to high quality, 'q' to quit")

# Function to toggle quality
def toggle_quality():
    current = layer.ray_tracing_resolution
    if current > 1.0:
        # Switch to high quality
        layer.ray_tracing_resolution = 0.001
        layer.name = 'cube_high_quality'
        print("\nSwitched to HIGH quality (ray_tracing_resolution=0.001)")
        print("Edges should now appear smooth")
    else:
        # Switch to low quality
        layer.ray_tracing_resolution = 4.0
        layer.name = 'cube_low_quality'
        print("\nSwitched to LOW quality (ray_tracing_resolution=4.0)")
        print("Edges should now appear jagged")
    print(f"Current ray_tracing_resolution: {layer.ray_tracing_resolution}")

# Bind key to toggle quality
@viewer.bind_key('c')
def change_quality(viewer):
    toggle_quality()

# Also test with different values
@viewer.bind_key('1')
def set_very_low(viewer):
    layer.ray_tracing_resolution = 16.0
    print(f"\nSet to VERY LOW quality: {layer.ray_tracing_resolution}")

@viewer.bind_key('2')
def set_low(viewer):
    layer.ray_tracing_resolution = 4.0
    print(f"\nSet to LOW quality: {layer.ray_tracing_resolution}")

@viewer.bind_key('3')
def set_default(viewer):
    layer.ray_tracing_resolution = 0.8
    print(f"\nSet to DEFAULT quality: {layer.ray_tracing_resolution}")

@viewer.bind_key('4')
def set_high(viewer):
    layer.ray_tracing_resolution = 0.1
    print(f"\nSet to HIGH quality: {layer.ray_tracing_resolution}")

@viewer.bind_key('5')
def set_very_high(viewer):
    layer.ray_tracing_resolution = 0.001
    print(f"\nSet to VERY HIGH quality: {layer.ray_tracing_resolution}")

print("\nKeyboard shortcuts:")
print("  c - Toggle between low (4.0) and high (0.001) quality")
print("  1 - Very low quality (16.0)")
print("  2 - Low quality (4.0)")
print("  3 - Default quality (0.8)")
print("  4 - High quality (0.1)")
print("  5 - Very high quality (0.001)")
print("  q - Quit")

# Run the napari event loop
napari.run()
