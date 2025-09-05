"""Tests for QtRayTracingSliderControl widget."""

import numpy as np
import pytest

from napari._qt.layer_controls.widgets.qt_ray_tracing_slider import (
    QtRayTracingSliderControl,
)
from napari.layers import Image, Labels


@pytest.mark.parametrize('layer_class', [Image, Labels])
def test_qt_ray_tracing_slider_creation(qtbot, layer_class):
    """Test that the slider control can be created for both layer types."""
    data = np.random.random((10, 10, 10))
    if layer_class == Labels:
        data = (data > 0.5).astype(np.uint8)

    layer = layer_class(data)
    control = QtRayTracingSliderControl(None, layer)
    # Don't add control itself to qtbot (it's QObject, not QWidget)
    # Add the actual widgets instead
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    assert control.ray_tracing_slider is not None
    assert control.ray_tracing_slider_label is not None


def test_qt_ray_tracing_slider_preset_values(qtbot):
    """Test that slider uses correct preset values."""
    data = np.random.random((10, 10, 10))
    layer = Image(data)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    # Check preset values
    expected_presets = [
        0.0001,
        0.001,
        0.01,
        0.1,
        0.8,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
    ]
    assert expected_presets == control.PRESET_VALUES

    # Slider should use indices
    assert control.ray_tracing_slider.minimum() == 0
    assert control.ray_tracing_slider.maximum() == len(expected_presets) - 1


def test_qt_ray_tracing_slider_discrete_movement(qtbot):
    """Test that slider moves between discrete preset values."""
    data = np.random.random((10, 10, 10))
    layer = Image(data, ray_tracing_step_size=0.8)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    # Initial value should be at index 4 (0.8)
    assert control.ray_tracing_slider.value() == 4
    assert layer.ray_tracing_step_size == 0.8

    # Move slider to different positions
    control.ray_tracing_slider.setValue(0)  # Should set to 0.0001
    assert layer.ray_tracing_step_size == 0.0001

    control.ray_tracing_slider.setValue(6)  # Should set to 2.0
    assert layer.ray_tracing_step_size == 2.0

    control.ray_tracing_slider.setValue(10)  # Should set to 32.0
    assert layer.ray_tracing_step_size == 32.0


def test_qt_ray_tracing_slider_label_display(qtbot):
    """Test that slider label shows correct formatted values."""
    data = np.random.random((10, 10, 10))
    layer = Image(data)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    # Test different value formats
    control.ray_tracing_slider.setValue(0)  # 0.0001
    assert '0.0001' in control.ray_tracing_slider._label.text()

    control.ray_tracing_slider.setValue(2)  # 0.01
    assert '0.01' in control.ray_tracing_slider._label.text()

    control.ray_tracing_slider.setValue(5)  # 1.0
    assert '1.0' in control.ray_tracing_slider._label.text()

    control.ray_tracing_slider.setValue(9)  # 16.0
    assert '16.0' in control.ray_tracing_slider._label.text()


def test_qt_ray_tracing_slider_label_editing(qtbot):
    """Test that editing the label allows custom values."""
    data = np.random.random((10, 10, 10))
    layer = Image(data)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    # Simulate entering a custom value
    control.ray_tracing_slider._label.setText('0.5')
    control._on_label_edited()

    # Layer should have the custom value
    assert layer.ray_tracing_step_size == 0.5

    # Slider should snap to nearest preset (index 3 = 0.1 or index 4 = 0.8)
    # Based on logarithmic distance, 0.5 is closer to 0.8
    assert control.ray_tracing_slider.value() == 4

    # But label should still show the custom value
    assert '0.5' in control.ray_tracing_slider._label.text()


def test_qt_ray_tracing_slider_invalid_label_input(qtbot):
    """Test that invalid label input is handled correctly."""
    data = np.random.random((10, 10, 10))
    layer = Image(data, ray_tracing_step_size=1.0)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    original_value = layer.ray_tracing_step_size

    # Try entering invalid text
    control.ray_tracing_slider._label.setText('invalid')
    control._on_label_edited()

    # Value should not change
    assert layer.ray_tracing_step_size == original_value

    # Try entering negative value
    control.ray_tracing_slider._label.setText('-1.0')
    control._on_label_edited()

    # Value should not change
    assert layer.ray_tracing_step_size == original_value

    # Try entering zero
    control.ray_tracing_slider._label.setText('0')
    control._on_label_edited()

    # Value should not change
    assert layer.ray_tracing_step_size == original_value


def test_qt_ray_tracing_slider_layer_sync(qtbot):
    """Test that slider syncs with layer value changes."""
    data = np.random.random((10, 10, 10))
    layer = Image(data)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    # Change layer value programmatically
    layer.ray_tracing_step_size = 0.001

    # Slider should update to nearest preset index (1)
    assert control.ray_tracing_slider.value() == 1

    # Change to a custom value
    layer.ray_tracing_step_size = 0.3

    # Slider should snap to nearest (probably index 3 or 4)
    # But label should show the actual value
    assert (
        '0.3' in control.ray_tracing_slider._label.text()
        or '0.300' in control.ray_tracing_slider._label.text()
    )


def test_qt_ray_tracing_slider_visibility(qtbot):
    """Test that slider visibility methods work correctly."""
    data = np.random.random((10, 10, 10))
    layer = Image(data)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    # Show the widgets first (they're not visible by default without a parent)
    control.ray_tracing_slider.show()
    control.ray_tracing_slider_label.show()

    # Now they should be visible
    assert control.ray_tracing_slider.isVisible()
    assert control.ray_tracing_slider_label.isVisible()

    # Hide the control
    control._on_display_change_hide()
    assert not control.ray_tracing_slider.isVisible()
    assert not control.ray_tracing_slider_label.isVisible()

    # Show the control
    control._on_display_change_show()
    assert control.ray_tracing_slider.isVisible()
    assert control.ray_tracing_slider_label.isVisible()


def test_qt_ray_tracing_slider_widget_controls(qtbot):
    """Test that get_widget_controls returns correct widgets."""
    data = np.random.random((10, 10, 10))
    layer = Image(data)
    control = QtRayTracingSliderControl(None, layer)
    qtbot.addWidget(control.ray_tracing_slider)
    qtbot.addWidget(control.ray_tracing_slider_label)

    controls = control.get_widget_controls()
    assert len(controls) == 1
    assert controls[0][0] == control.ray_tracing_slider_label
    assert controls[0][1] == control.ray_tracing_slider
