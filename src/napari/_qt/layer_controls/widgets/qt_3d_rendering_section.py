"""Qt widget for grouping 3D rendering controls."""

from qtpy.QtWidgets import QLabel

from napari.utils.translations import trans


class Qt3DRenderingSection:
    """Helper class to manage 3D rendering controls with a section label."""

    def __init__(self, parent_layout):
        """Initialize the section.

        Parameters
        ----------
        parent_layout : QFormLayout
            The parent form layout to add controls to
        """
        self.parent_layout = parent_layout
        self._controls = []
        self._widgets = []
        self.setup_section()

    def setup_section(self):
        """Add the section label to the parent layout."""
        # Simple section label, no special styling
        self.section_label = QLabel(trans._('3D rendering:'))
        # Add label spanning both columns
        self.parent_layout.addRow(self.section_label)
        self._widgets.append(self.section_label)

    def add_control(self, label_widget, control_widget):
        """Add a control to the parent layout under this section.

        Parameters
        ----------
        label_widget : QWidget
            The label widget for the control
        control_widget : QWidget
            The actual control widget
        """
        # Add the control directly to the parent layout
        self.parent_layout.addRow(label_widget, control_widget)
        self._controls.append((label_widget, control_widget))

    def show_section(self):
        """Show the 3D rendering section label."""
        # Only show section widgets (the label), not the controls
        # Let controls manage their own visibility
        for widget in self._widgets:
            widget.setVisible(True)

    def hide_section(self):
        """Hide the 3D rendering section and all its controls."""
        # Hide section widgets
        for widget in self._widgets:
            widget.setVisible(False)
        # Hide all controls when switching to 2D
        for label, control in self._controls:
            label.setVisible(False)
            control.setVisible(False)
