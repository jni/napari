from typing import ClassVar

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

        # Setup widgets
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)

        # Set up the slider with actual float values
        # Use min and max from preset values
        sld.setMinimum(self.PRESET_VALUES[0])
        sld.setMaximum(self.PRESET_VALUES[-1])
        sld.setSingleStep(0.1)
        sld.setPageStep(1.0)

        # Set the current value
        current_value = self._layer.ray_tracing_resolution
        sld.setValue(current_value)

        # Connect the value change to our custom handler
        sld.valueChanged.connect(self._on_slider_change)
        self.ray_tracing_slider = sld

        self.ray_tracing_slider_label = QtWrappedLabel(
            trans._('ray tracing resolution:')
        )

    def _find_closest_preset_value(self, value: float) -> float:
        """Find the preset value closest to the given value."""
        import numpy as np

        distances = [
            abs(np.log10(value / preset)) if preset > 0 else float('inf')
            for preset in self.PRESET_VALUES
        ]
        min_index = distances.index(min(distances))
        return self.PRESET_VALUES[min_index]

    def _on_slider_change(self, value: float):
        """Handle slider value changes and snap to nearest preset."""
        # Find the closest preset value
        closest_preset = self._find_closest_preset_value(value)
        # Only update if it's different from current value to avoid loops
        if abs(self._layer.ray_tracing_resolution - closest_preset) > 1e-6:
            self._layer.ray_tracing_resolution = closest_preset

    def _on_ray_tracing_resolution_change(self):
        """Receive the layer model ray_tracing_resolution change event and update the slider."""
        with qt_signals_blocked(self.ray_tracing_slider):
            current_value = self._layer.ray_tracing_resolution
            self.ray_tracing_slider.setValue(current_value)

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
