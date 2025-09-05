from typing import ClassVar

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtRayTracingSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the ray tracing
    resolution attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer (Image or Labels).

    Attributes
    ----------
    ray_tracing_slider : superqt.QLabeledDoubleSlider
        Ray tracing resolution adjustment slider widget.
    ray_tracing_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the ray tracing resolution slider widget.
    """

    # Define the preset values for the slider
    PRESET_VALUES: ClassVar[list[float]] = [
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

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.ray_tracing_resolution.connect(
            self._on_ray_tracing_resolution_change
        )

        # Setup widgets - use QLabeledDoubleSlider but configure it for discrete values
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)

        # Use indices for the slider (0 to len-1) and map to actual values
        sld.setMinimum(0)
        sld.setMaximum(len(self.PRESET_VALUES) - 1)
        sld.setSingleStep(1)
        sld.setPageStep(1)
        sld.setDecimals(0)  # Show as integer indices

        # Find the closest preset index for the current value
        current_value = self._layer.ray_tracing_resolution
        current_index = self._find_closest_preset_index(current_value)
        sld.setValue(current_index)

        # Override the label to show the actual value
        self._update_slider_label(sld, current_index)

        # Connect the value change to our custom handler
        sld.valueChanged.connect(self._on_slider_change)

        # Also handle direct label editing
        sld._label.editingFinished.connect(self._on_label_edited)

        self.ray_tracing_slider = sld
        self.ray_tracing_slider_label = QtWrappedLabel(
            trans._('ray tracing resolution:')
        )

    def _find_closest_preset_index(self, value: float) -> int:
        """Find the index of the preset value closest to the given value."""
        # Use logarithmic distance for better perceptual spacing
        distances = [
            abs(np.log10(value / preset)) if preset > 0 else float('inf')
            for preset in self.PRESET_VALUES
        ]
        return int(distances.index(min(distances)))

    def _update_slider_label(self, slider: QLabeledDoubleSlider, index: int):
        """Update the slider label to show the actual preset value."""
        if 0 <= index < len(self.PRESET_VALUES):
            actual_value = self.PRESET_VALUES[index]
            # Format the label based on value magnitude
            if actual_value >= 1.0:
                label_text = f'{actual_value:.1f}'
            elif actual_value >= 0.01:
                label_text = f'{actual_value:.3f}'
            else:
                label_text = f'{actual_value:.4f}'
            slider._label.setText(label_text)

    def _on_slider_change(self, index_value: float):
        """Handle slider value changes (index-based)."""
        index = round(index_value)
        if 0 <= index < len(self.PRESET_VALUES):
            # Update the label to show the actual value
            self._update_slider_label(self.ray_tracing_slider, index)
            # Set the layer value
            new_value = self.PRESET_VALUES[index]
            if abs(self._layer.ray_tracing_resolution - new_value) > 1e-9:
                self._layer.ray_tracing_resolution = new_value

    def _on_label_edited(self):
        """Handle direct editing of the label value."""
        try:
            # Get the value entered by the user
            text = self.ray_tracing_slider._label.text()
            user_value = float(text)

            if user_value <= 0:
                # Invalid value, reset to current
                current_index = self._find_closest_preset_index(
                    self._layer.ray_tracing_resolution
                )
                self._update_slider_label(
                    self.ray_tracing_slider, current_index
                )
                return

            # Set the layer value directly (user can enter any positive value)
            self._layer.ray_tracing_resolution = user_value

            # Find closest preset and update slider position
            closest_index = self._find_closest_preset_index(user_value)
            with qt_signals_blocked(self.ray_tracing_slider):
                self.ray_tracing_slider.setValue(closest_index)
                # Keep the user's entered value in the label
                self.ray_tracing_slider._label.setText(text)
        except (ValueError, AttributeError):
            # Invalid input, reset to current value
            current_index = self._find_closest_preset_index(
                self._layer.ray_tracing_resolution
            )
            self._update_slider_label(self.ray_tracing_slider, current_index)

    def _on_ray_tracing_resolution_change(self):
        """Receive the layer model ray_tracing_resolution change event and update the slider."""
        with qt_signals_blocked(self.ray_tracing_slider):
            current_value = self._layer.ray_tracing_resolution
            # Find closest preset index
            closest_index = self._find_closest_preset_index(current_value)
            self.ray_tracing_slider.setValue(closest_index)

            # If the actual value is not a preset, show it in the label
            if abs(current_value - self.PRESET_VALUES[closest_index]) > 1e-9:
                # User has set a custom value programmatically
                if current_value >= 1.0:
                    label_text = f'{current_value:.1f}'
                elif current_value >= 0.01:
                    label_text = f'{current_value:.3f}'
                else:
                    label_text = f'{current_value:.4f}'
                self.ray_tracing_slider._label.setText(label_text)
            else:
                # It's a preset value, update normally
                self._update_slider_label(
                    self.ray_tracing_slider, closest_index
                )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.ray_tracing_slider_label, self.ray_tracing_slider)]

    def _on_display_change_hide(self):
        """Hide the control when switching to 2D view."""
        self.ray_tracing_slider.hide()
        self.ray_tracing_slider_label.hide()

    def _on_display_change_show(self):
        """Show the control when switching to 3D view."""
        self.ray_tracing_slider.show()
        self.ray_tracing_slider_label.show()
