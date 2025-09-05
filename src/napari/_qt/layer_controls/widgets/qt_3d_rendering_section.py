"""Qt widget for grouping 3D rendering controls."""

from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from napari.utils.translations import trans


class Qt3DRenderingSection(QWidget):
    """Container widget for 3D rendering controls with a section label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._controls = []

    def setup_ui(self):
        """Set up the UI with a section label and container for controls."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(4)

        # Section header
        self.section_label = QLabel(trans._('3D rendering'))
        self.section_label.setStyleSheet(
            'font-weight: bold; margin-bottom: 4px;'
        )
        layout.addWidget(self.section_label)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Container for the controls
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout()
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(4)
        self.controls_container.setLayout(self.controls_layout)
        layout.addWidget(self.controls_container)

        self.setLayout(layout)

    def add_control(self, label_widget, control_widget):
        """Add a control to the 3D rendering section.

        Parameters
        ----------
        label_widget : QWidget
            The label widget for the control
        control_widget : QWidget
            The actual control widget
        """
        # Create horizontal container for label and control
        from qtpy.QtWidgets import QHBoxLayout, QWidget

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(label_widget)
        row_layout.addWidget(control_widget)
        row.setLayout(row_layout)

        self.controls_layout.addWidget(row)
        self._controls.append((label_widget, control_widget))

    def set_visible(self, visible):
        """Set visibility of the entire section."""
        self.setVisible(visible)

    def show_section(self):
        """Show the 3D rendering section."""
        self.setVisible(True)

    def hide_section(self):
        """Hide the 3D rendering section."""
        self.setVisible(False)
