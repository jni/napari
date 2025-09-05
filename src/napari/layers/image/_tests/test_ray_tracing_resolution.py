"""Tests for ray_tracing_resolution parameter in Image and Labels layers."""

import numpy as np
import pytest

from napari.layers import Image, Labels


class TestRayTracingResolution:
    """Test suite for ray_tracing_resolution parameter."""

    def test_image_layer_default_value(self):
        """Test that Image layer has correct default ray_tracing_resolution."""
        data = np.random.random((10, 10, 10))
        layer = Image(data)
        assert layer.ray_tracing_resolution == 0.8

    def test_image_layer_custom_value(self):
        """Test setting custom ray_tracing_resolution on Image layer."""
        data = np.random.random((10, 10, 10))
        layer = Image(data, ray_tracing_resolution=0.1)
        assert layer.ray_tracing_resolution == 0.1

    def test_image_layer_value_change(self):
        """Test changing ray_tracing_resolution after layer creation."""
        data = np.random.random((10, 10, 10))
        layer = Image(data)

        # Test setting various values
        test_values = [0.001, 0.1, 1.0, 4.0, 16.0]
        for value in test_values:
            layer.ray_tracing_resolution = value
            assert layer.ray_tracing_resolution == value

    def test_image_layer_invalid_values(self):
        """Test that invalid ray_tracing_resolution values raise errors."""
        data = np.random.random((10, 10, 10))

        # Test negative value
        with pytest.raises(
            ValueError, match='ray_tracing_resolution must be positive'
        ):
            Image(data, ray_tracing_resolution=-1.0)

        # Test zero value
        with pytest.raises(
            ValueError, match='ray_tracing_resolution must be positive'
        ):
            Image(data, ray_tracing_resolution=0.0)

        # Test setting invalid value on existing layer
        layer = Image(data)
        with pytest.raises(
            ValueError, match='ray_tracing_resolution must be positive'
        ):
            layer.ray_tracing_resolution = -0.5

    def test_labels_layer_default_value(self):
        """Test that Labels layer has correct default ray_tracing_resolution."""
        data = np.random.randint(0, 10, (10, 10, 10), dtype=np.uint8)
        layer = Labels(data)
        assert layer.ray_tracing_resolution == 0.8

    def test_labels_layer_custom_value(self):
        """Test setting custom ray_tracing_resolution on Labels layer."""
        data = np.random.randint(0, 10, (10, 10, 10), dtype=np.uint8)
        layer = Labels(data, ray_tracing_resolution=2.0)
        assert layer.ray_tracing_resolution == 2.0

    def test_labels_layer_value_change(self):
        """Test changing ray_tracing_resolution after layer creation."""
        data = np.random.randint(0, 10, (10, 10, 10), dtype=np.uint8)
        layer = Labels(data)

        # Test setting various values
        test_values = [0.0001, 0.01, 0.8, 8.0, 32.0]
        for value in test_values:
            layer.ray_tracing_resolution = value
            assert layer.ray_tracing_resolution == value

    def test_labels_layer_invalid_values(self):
        """Test that invalid ray_tracing_resolution values raise errors."""
        data = np.random.randint(0, 10, (10, 10, 10), dtype=np.uint8)

        # Test negative value
        with pytest.raises(
            ValueError, match='ray_tracing_resolution must be positive'
        ):
            Labels(data, ray_tracing_resolution=-2.0)

        # Test zero value
        with pytest.raises(
            ValueError, match='ray_tracing_resolution must be positive'
        ):
            Labels(data, ray_tracing_resolution=0.0)

        # Test setting invalid value on existing layer
        layer = Labels(data)
        with pytest.raises(
            ValueError, match='ray_tracing_resolution must be positive'
        ):
            layer.ray_tracing_resolution = 0.0

    def test_ray_tracing_resolution_event(self):
        """Test that changing ray_tracing_resolution emits proper event."""
        data = np.random.random((10, 10, 10))
        layer = Image(data)

        # Track event emissions
        event_count = 0
        event_values = []

        def on_ray_tracing_change(event):
            nonlocal event_count, event_values
            event_count += 1
            event_values.append(layer.ray_tracing_resolution)

        layer.events.ray_tracing_resolution.connect(on_ray_tracing_change)

        # Change the value
        layer.ray_tracing_resolution = 0.1
        assert event_count == 1
        assert event_values[-1] == 0.1

        layer.ray_tracing_resolution = 4.0
        assert event_count == 2
        assert event_values[-1] == 4.0

    def test_image_layer_state_dict(self):
        """Test that ray_tracing_resolution is included in layer state."""
        data = np.random.random((10, 10, 10))
        layer = Image(data, ray_tracing_resolution=0.01)

        state = layer._get_state()
        assert 'ray_tracing_resolution' in state
        assert state['ray_tracing_resolution'] == 0.01

    def test_value_persistence(self):
        """Test that ray_tracing_resolution value persists through operations."""
        data = np.random.random((10, 10, 10))
        layer = Image(data, ray_tracing_resolution=0.5)

        # Value should persist through other property changes
        layer.opacity = 0.5
        assert layer.ray_tracing_resolution == 0.5

        layer.colormap = 'viridis'
        assert layer.ray_tracing_resolution == 0.5

        layer.rendering = 'iso'
        assert layer.ray_tracing_resolution == 0.5
