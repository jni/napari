"""Tests for ray_tracing_resolution integration with VisPy visuals."""

import numpy as np

from napari._vispy.layers.image import VispyImageLayer
from napari._vispy.layers.labels import VispyLabelsLayer
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.layers import Image, Labels


def test_vispy_image_layer_ray_tracing_connection():
    """Test that ray_tracing_resolution connects to VolumeNode in Image layer."""
    # Create 3D data to ensure VolumeNode is used
    data = np.random.random((10, 10, 10))
    layer = Image(data, ray_tracing_resolution=0.5)

    # Create VisPy layer
    vispy_layer = VispyImageLayer(layer)

    # Force 3D display by updating slice input
    from napari.layers.utils._slice_input import _SliceInput

    layer._slice_input = _SliceInput(
        ndisplay=3,
        world_slice=layer._slice_input.world_slice,
        order=layer._slice_input.order,
    )
    vispy_layer._on_display_change()
    vispy_layer.reset()

    # In 3D mode, must use VolumeNode
    node = vispy_layer.node
    assert isinstance(node, VolumeNode), (
        f'Expected VolumeNode, got {type(node)}'
    )

    # Check that relative_step_size is set correctly
    assert node.relative_step_size == 0.5

    # Change the layer value
    layer.ray_tracing_resolution = 2.0

    # VolumeNode should update
    assert node.relative_step_size == 2.0


def test_vispy_labels_layer_ray_tracing_connection():
    """Test that ray_tracing_resolution connects to VolumeNode in Labels layer."""
    # Create 3D data to ensure VolumeNode is used
    data = np.random.randint(0, 10, (10, 10, 10), dtype=np.uint8)
    layer = Labels(data, ray_tracing_resolution=1.5)

    # Create VisPy layer
    vispy_layer = VispyLabelsLayer(layer)

    # Force 3D display by updating slice input
    from napari.layers.utils._slice_input import _SliceInput

    layer._slice_input = _SliceInput(
        ndisplay=3,
        world_slice=layer._slice_input.world_slice,
        order=layer._slice_input.order,
    )
    vispy_layer._on_display_change()
    vispy_layer.reset()

    # In 3D mode, must use VolumeNode
    node = vispy_layer.node
    assert isinstance(node, VolumeNode), (
        f'Expected VolumeNode, got {type(node)}'
    )

    # Check that relative_step_size is set correctly
    assert node.relative_step_size == 1.5

    # Change the layer value
    layer.ray_tracing_resolution = 0.1

    # VolumeNode should update
    assert node.relative_step_size == 0.1


def test_vispy_ray_tracing_event_connection():
    """Test that ray_tracing_resolution events are properly connected."""
    # Create 3D data
    data = np.random.random((10, 10, 10))
    layer = Image(data)

    vispy_layer = VispyImageLayer(layer)

    # Check that event is connected
    assert hasattr(layer.events, 'ray_tracing_resolution')

    # Test that changing value triggers update
    event_called = False

    def mock_handler():
        nonlocal event_called
        event_called = True

    # Replace the handler temporarily
    original_handler = vispy_layer._on_ray_tracing_resolution_change
    vispy_layer._on_ray_tracing_resolution_change = mock_handler

    layer.ray_tracing_resolution = 0.2
    assert event_called

    # Restore original handler
    vispy_layer._on_ray_tracing_resolution_change = original_handler


def test_vispy_reset_includes_ray_tracing():
    """Test that reset() method updates ray_tracing_resolution."""
    # Create 3D data to ensure VolumeNode is used
    data = np.random.random((10, 10, 10))
    layer = Image(data, ray_tracing_resolution=0.01)

    vispy_layer = VispyImageLayer(layer)

    # Force 3D display by updating slice input
    from napari.layers.utils._slice_input import _SliceInput

    layer._slice_input = _SliceInput(
        ndisplay=3,
        world_slice=layer._slice_input.world_slice,
        order=layer._slice_input.order,
    )
    vispy_layer._on_display_change()
    vispy_layer.reset()

    # Change the value
    layer.ray_tracing_resolution = 8.0

    # Reset should apply the new value
    vispy_layer.reset()

    node = vispy_layer.node
    assert isinstance(node, VolumeNode), (
        f'Expected VolumeNode, got {type(node)}'
    )
    assert node.relative_step_size == 8.0


def test_vispy_2d_mode_no_effect():
    """Test that ray_tracing_resolution has no effect in 2D mode."""
    # Create 2D data
    data = np.random.random((10, 10))
    layer = Image(data, ray_tracing_resolution=0.1)

    vispy_layer = VispyImageLayer(layer)

    # 2D data will automatically use 2D rendering
    # Just ensure we reset to apply settings
    vispy_layer._on_display_change()
    vispy_layer.reset()

    # In 2D mode, node should not be VolumeNode (should be ImageNode)
    node = vispy_layer.node
    assert not isinstance(node, VolumeNode), (
        f'Expected ImageNode in 2D, got {type(node)}'
    )

    # Changing value should not cause errors
    layer.ray_tracing_resolution = 4.0  # Should not raise


def test_vispy_different_values_for_layers():
    """Test that different layers can have different ray_tracing_resolution values."""
    # Create 3D data to ensure VolumeNode is used
    data1 = np.random.random((10, 10, 10))
    data2 = np.random.randint(0, 10, (10, 10, 10), dtype=np.uint8)

    image_layer = Image(data1, ray_tracing_resolution=0.1)
    labels_layer = Labels(data2, ray_tracing_resolution=4.0)

    vispy_image = VispyImageLayer(image_layer)
    vispy_labels = VispyLabelsLayer(labels_layer)

    # Force 3D display for both layers
    from napari.layers.utils._slice_input import _SliceInput

    image_layer._slice_input = _SliceInput(
        ndisplay=3,
        world_slice=image_layer._slice_input.world_slice,
        order=image_layer._slice_input.order,
    )
    vispy_image._on_display_change()
    vispy_image.reset()

    labels_layer._slice_input = _SliceInput(
        ndisplay=3,
        world_slice=labels_layer._slice_input.world_slice,
        order=labels_layer._slice_input.order,
    )
    vispy_labels._on_display_change()
    vispy_labels.reset()

    # Both should be VolumeNodes in 3D
    assert isinstance(vispy_image.node, VolumeNode), (
        f'Expected VolumeNode for image, got {type(vispy_image.node)}'
    )
    assert isinstance(vispy_labels.node, VolumeNode), (
        f'Expected VolumeNode for labels, got {type(vispy_labels.node)}'
    )

    # Each should maintain its own value
    assert vispy_image.node.relative_step_size == 0.1
    assert vispy_labels.node.relative_step_size == 4.0

    # Changing one should not affect the other
    image_layer.ray_tracing_resolution = 1.0
    assert vispy_image.node.relative_step_size == 1.0
    assert vispy_labels.node.relative_step_size == 4.0  # Unchanged
