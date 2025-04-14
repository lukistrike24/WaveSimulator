# wave_gui.py
# WARNING: All explicit error handling has been removed for brevity.
# This code is fragile and not suitable for production use.
import sys
import numpy as np
import time
import math

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QFormLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QComboBox, QFrame, QTextEdit, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPixmapItem
)
from PySide6.QtGui import (
    QPixmap, QPainter, QColor, QPen, QBrush, QPalette, QCursor, QImage, qRgb, QTransform
)
from PySide6.QtCore import Qt, QMimeData, QRectF, QPointF, QThread, Signal, Slot

# --- Import Simulator Logic (Assume it exists - UNCHANGED) ---
from wave_simulator import SimulatorLogic, DEFAULT_NX, DEFAULT_NY, DEFAULT_DX

# --- GUI Constants ---
SCENE_SIZE = 500  # Pixel size for the main scene/view


# --- Colormap Generation ---
def generate_colormap(cmap_name='seismic', n_colors=256):
    """Generates a simple RGB colormap as a NumPy array."""
    colors = np.zeros((n_colors, 3), dtype=np.uint8)
    if cmap_name == 'seismic':  # Simple Blue -> White -> Red
        for i in range(n_colors):
            val = i / (n_colors - 1);
            g = 0;
            b = 0;
            r = 0
            if val < 0.5:
                g = int(255 * (val / 0.5));
                b = 255;
                r = g
            else:
                g = int(255 * ((1.0 - val) / 0.5));
                r = 255;
                b = g
            colors[i] = [r, g, b]
    elif cmap_name == 'grayscale':
        for i in range(n_colors): val = int(255 * i / (n_colors - 1)); colors[i] = [val, val, val]
    else:  # Default grayscale
        print(f"Warning: Colormap '{cmap_name}' not recognized, using grayscale.")
        return generate_colormap('grayscale', n_colors)
    return colors


# Precompute colormap
COLORMAP = generate_colormap('seismic')
COLORMAP_SIZE = COLORMAP.shape[0]


def numpy_to_qimage(data: np.ndarray, vmin: float, vmax: float, colormap: np.ndarray):
    """Converts a 2D NumPy float array to a QImage using a colormap (no error checks)."""
    height, width = data.shape
    q_image = QImage(width, height, QImage.Format_RGB32)
    # Ensure vmin and vmax are different to avoid division by zero
    if vmax <= vmin: vmax = vmin + 1e-6  # Add a small epsilon
    norm_data = (data - vmin) / (vmax - vmin)
    indices = np.clip(norm_data * (COLORMAP_SIZE - 1), 0, COLORMAP_SIZE - 1).astype(np.uint8)
    ptr = q_image.bits()
    # Assume buffer size is correct
    arr = np.asarray(ptr).reshape(height, width, 4)  # Assume reshape works
    rgb_values = colormap[indices]
    # Fill the QImage buffer (BGRA order for Format_RGB32 on little-endian)
    arr[:, :, 0] = rgb_values[:, :, 2]  # Blue
    arr[:, :, 1] = rgb_values[:, :, 1]  # Green
    arr[:, :, 2] = rgb_values[:, :, 0]  # Red
    arr[:, :, 3] = 255  # Alpha
    return q_image


# --- Simulation Thread (Simplified Error Handling - UNCHANGED) ---
class SimulationThread(QThread):
    frameReady = Signal(np.ndarray, float, int)
    statusUpdate = Signal(str)
    simulationFinished = Signal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._simulator_logic = None

    def run(self):
        self._simulator_logic = SimulatorLogic(config=self.config,
                                               status_callback=self.statusUpdate.emit,
                                               frame_callback=self.frameReady.emit)
        self._simulator_logic.run()
        self.simulationFinished.emit()

    def stop_simulation(self):
        if self._simulator_logic: self._simulator_logic.stop()


# --- Draggable Frame ---
class DraggableControlFrame(QFrame):
    initiateDrag = Signal(str, dict)

    def __init__(self, item_type_func, param_func, parent=None):
        super().__init__(parent)
        self.item_type_func = item_type_func
        self.param_func = param_func
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Configure then Click-Drag onto the simulation area")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item_type = self.item_type_func()
            params = self.param_func(item_type)
            self.initiateDrag.emit(item_type, params)
        else:
            super().mousePressEvent(event)


# --- Simulation View ---
class SimulationView(QGraphicsView):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMinimumSize(SCENE_SIZE + 2, SCENE_SIZE + 2)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setRenderHint(QPainter.Antialiasing)  # Improve rendering quality

    def set_scene_size(self, nx, ny, dx):
        width, height = nx * dx, ny * dx
        self.scene.setSceneRect(0, 0, width, height)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)


# --- MainWindow Class (Simplified) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wave Simulator GUI (PySide - Simplified)")
        self.setGeometry(50, 50, 950, 650)

        self.simulation_thread = None
        self.placed_items_data = []
        self.config_items_gfx = []
        self._is_dragging_config = False
        self._drag_item_type = None
        self._drag_item_params = None
        self._drag_visual_item = None

        # --- Central Widget & Layout ---
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QHBoxLayout(self.centralWidget)

        # --- Left Panel: Controls ---
        self.controlPanel = QWidget()
        self.controlPanel.setMaximumWidth(350)  # Increased width slightly
        self.controlLayout = QVBoxLayout(self.controlPanel)
        self.mainLayout.addWidget(self.controlPanel, 1)

        # (Simulation Settings Group)
        simSettingsGroup = QGroupBox("Simulation Settings")
        self.simSettingsGroup = simSettingsGroup  # Keep reference for enabling/disabling
        simSettingsLayout = QFormLayout(simSettingsGroup)
        self.nxSpin = QSpinBox();
        self.nxSpin.setRange(1, 4096);
        self.nxSpin.setValue(DEFAULT_NX)
        self.nySpin = QSpinBox();
        self.nySpin.setRange(1, 4096);
        self.nySpin.setValue(DEFAULT_NY)
        self.stepsSpin = QSpinBox();
        self.stepsSpin.setRange(10, 100000);
        self.stepsSpin.setValue(20000)
        self.pmlSpin = QSpinBox();
        self.pmlSpin.setRange(0, 200);
        self.pmlSpin.setValue(150)
        self.courantSpin = QDoubleSpinBox();
        self.courantSpin.setRange(0.01, 1.0);
        self.courantSpin.setValue(0.1);
        self.courantSpin.setSingleStep(0.05)
        self.dampingSpin = QDoubleSpinBox();
        self.dampingSpin.setRange(0.0, 1000.0);
        self.dampingSpin.setValue(50.0)
        self.vizIntervalSpin = QSpinBox();
        self.vizIntervalSpin.setRange(1, 2000);
        self.vizIntervalSpin.setValue(50)
        self.vminSpin = QDoubleSpinBox();
        self.vminSpin.setRange(-100.0, 100.0);
        self.vminSpin.setValue(-0.1);
        self.vminSpin.setSingleStep(0.1);
        self.vminSpin.setDecimals(3)
        self.vmaxSpin = QDoubleSpinBox();
        self.vmaxSpin.setRange(-100.0, 100.0);
        self.vmaxSpin.setValue(0.1);
        self.vmaxSpin.setSingleStep(0.1);
        self.vmaxSpin.setDecimals(3)
        simSettingsLayout.addRow("Grid NX:", self.nxSpin)
        simSettingsLayout.addRow("Grid NY:", self.nySpin)
        simSettingsLayout.addRow("Max Steps:", self.stepsSpin)
        simSettingsLayout.addRow("PML Width:", self.pmlSpin)
        simSettingsLayout.addRow("Courant:", self.courantSpin)
        simSettingsLayout.addRow("Max Damping:", self.dampingSpin)
        simSettingsLayout.addRow("Vis Interval:", self.vizIntervalSpin)
        simSettingsLayout.addRow("Vis Min Amp:", self.vminSpin)
        simSettingsLayout.addRow("Vis Max Amp:", self.vmaxSpin)
        self.controlLayout.addWidget(simSettingsGroup)

        # (Building Block Selection Group)
        blockSelectGroup = QGroupBox("Add Building Block")
        self.blockSelectGroup = blockSelectGroup  # Keep reference
        blockSelectLayout = QVBoxLayout(blockSelectGroup)
        typeLayout = QHBoxLayout();
        typeLayout.addWidget(QLabel("Type:"))
        self.blockTypeCombo = QComboBox();
        # --- ADD 'Source Grid' to dropdown ---
        self.blockTypeCombo.addItems(["Source", "Obstacle", "Source Grid"])
        typeLayout.addWidget(self.blockTypeCombo);
        blockSelectLayout.addLayout(typeLayout)
        self.draggableControlFrame = DraggableControlFrame(self.get_current_block_type,
                                                           self.get_current_block_parameters)
        self.draggableLayout = QVBoxLayout(self.draggableControlFrame)
        self.draggableLayout.addWidget(QLabel("Configure && Click Here,\n then Drag onto Sim Area ->"))
        self.blockParamsLayout = QFormLayout()
        self.draggableLayout.addLayout(self.blockParamsLayout)
        blockSelectLayout.addWidget(self.draggableControlFrame)

        # --- Shared Source/Grid parameters ---
        self.sourceFreqSpin = QDoubleSpinBox();
        self.sourceFreqSpin.setRange(0.1, 2000.0);
        self.sourceFreqSpin.setValue(0.25)  # Default Freq
        self.sourceAmpSpin = QDoubleSpinBox();
        self.sourceAmpSpin.setRange(0.0, 20000.0);
        self.sourceAmpSpin.setValue(100.0)

        # --- Obstacle parameters ---
        self.obstacleWidthSpin = QSpinBox();
        self.obstacleWidthSpin.setRange(1, DEFAULT_NX);
        self.obstacleWidthSpin.setValue(10)
        self.obstacleHeightSpin = QSpinBox();
        self.obstacleHeightSpin.setRange(1, DEFAULT_NY);
        self.obstacleHeightSpin.setValue(50)

        # --- Source Grid parameters ---
        self.sourceGridNxSpin = QSpinBox();
        self.sourceGridNxSpin.setRange(1, 100);
        self.sourceGridNxSpin.setValue(5)
        self.sourceGridNySpin = QSpinBox();
        self.sourceGridNySpin.setRange(1, 100);
        self.sourceGridNySpin.setValue(5)
        self.sourceGridSpacingXSpin = QSpinBox();
        self.sourceGridSpacingXSpin.setRange(1, 100);
        self.sourceGridSpacingXSpin.setValue(10)
        self.sourceGridSpacingYSpin = QSpinBox();
        self.sourceGridSpacingYSpin.setRange(1, 100);
        self.sourceGridSpacingYSpin.setValue(10)

        # --- Add ALL parameter widgets to the layout initially ---
        # They will be shown/hidden by update_block_parameters_display
        self.blockParamsLayout.addRow("Frequency (Hz):", self.sourceFreqSpin)  # Used by Source & Source Grid
        self.blockParamsLayout.addRow("Amplitude:", self.sourceAmpSpin)  # Used by Source & Source Grid
        self.blockParamsLayout.addRow("Width (cells):", self.obstacleWidthSpin)  # Obstacle only
        self.blockParamsLayout.addRow("Height (cells):", self.obstacleHeightSpin)  # Obstacle only
        self.blockParamsLayout.addRow("Grid Count X:", self.sourceGridNxSpin)  # Source Grid only
        self.blockParamsLayout.addRow("Grid Count Y:", self.sourceGridNySpin)  # Source Grid only
        self.blockParamsLayout.addRow("Grid Spacing X:", self.sourceGridSpacingXSpin)  # Source Grid only
        self.blockParamsLayout.addRow("Grid Spacing Y:", self.sourceGridSpacingYSpin)  # Source Grid only

        self.controlLayout.addWidget(blockSelectGroup)

        # (Controls & Status Group)
        runGroup = QGroupBox("Control & Status")
        runLayout = QVBoxLayout(runGroup)
        self.clearConfigButton = QPushButton("Clear Configuration")
        runLayout.addWidget(self.clearConfigButton)
        runButtonLayout = QHBoxLayout()
        self.runButton = QPushButton("▶ Run");
        self.stopButton = QPushButton("■ Stop");
        self.stopButton.setEnabled(False)
        runButtonLayout.addWidget(self.runButton);
        runButtonLayout.addWidget(self.stopButton)
        runLayout.addLayout(runButtonLayout)
        runLayout.addWidget(QLabel("Status Log:"))
        self.statusLog = QTextEdit();
        self.statusLog.setReadOnly(True);
        self.statusLog.setMaximumHeight(100)
        runLayout.addWidget(self.statusLog)
        self.controlLayout.addWidget(runGroup)
        self.controlLayout.addStretch()

        # --- Right Panel: Combined View ---
        self.simulationView = SimulationView(self)
        self.mainLayout.addWidget(self.simulationView, 3)

        # --- Initialize Scene & Connect Signals ---
        self.init_scene()
        self.blockTypeCombo.currentIndexChanged.connect(self.update_block_parameters_display)
        self.runButton.clicked.connect(self.start_simulation)
        self.stopButton.clicked.connect(self.stop_simulation)
        self.clearConfigButton.clicked.connect(self.clear_configuration)
        self.nxSpin.valueChanged.connect(self._handle_grid_change)
        self.nySpin.valueChanged.connect(self._handle_grid_change)
        self.draggableControlFrame.initiateDrag.connect(self._start_canvas_drag_mode)
        self.simulationView.mousePressEvent = self._view_mouse_press
        self.simulationView.mouseMoveEvent = self._view_mouse_move
        self.simulationView.mouseReleaseEvent = self._view_mouse_release
        self.update_block_parameters_display()  # Call initially to set visibility

    def _handle_grid_change(self):
        nx, ny = self.nxSpin.value(), self.nySpin.value()
        self.obstacleWidthSpin.setRange(1, nx)
        self.obstacleHeightSpin.setRange(1, ny)
        # Optionally, adjust source grid ranges based on sim size too?
        # self.sourceGridNxSpin.setRange(1, nx // self.sourceGridSpacingXSpin.value()) # Needs care
        # self.sourceGridNySpin.setRange(1, ny // self.sourceGridSpacingYSpin.value()) # Needs care
        self.init_scene()

    def init_scene(self):
        """Sets up the QGraphicsScene with background and placeholder."""
        nx = self.nxSpin.value()
        ny = self.nySpin.value()
        dx = DEFAULT_DX
        scene = self.simulationView.scene
        scene.clear()
        self.config_items_gfx.clear()

        self.simulationView.set_scene_size(nx, ny, dx)

        if nx > 0 and ny > 0:
            initial_image = QImage(nx, ny, QImage.Format_RGB32)
            initial_image.fill(Qt.gray)
            initial_pixmap = QPixmap.fromImage(initial_image)
            self.wavePixmapItem = QGraphicsPixmapItem(initial_pixmap)

            scale_factor = dx
            transform = QTransform()
            transform.scale(scale_factor, scale_factor)
            self.wavePixmapItem.setTransform(transform)

            self.wavePixmapItem.setPos(0, 0)
            self.wavePixmapItem.setZValue(-1)
            scene.addItem(self.wavePixmapItem)
        else:
            self.wavePixmapItem = None  # Handle case of 0 size grid

        self.draw_configuration_overlay()

    def get_current_block_type(self):
        # Use a consistent internal name, e.g., 'source_grid'
        text = self.blockTypeCombo.currentText()
        if text == "Source":
            return 'source'
        elif text == "Obstacle":
            return 'obstacle'
        elif text == "Source Grid":
            return 'source_grid'  # Keep temporary type for GUI logic
        return 'unknown'  # Should not happen

    def update_block_parameters_display(self):
        block_type = self.get_current_block_type()

        is_src = (block_type == 'source')
        is_obs = (block_type == 'obstacle')
        is_grid = (block_type == 'source_grid')

        # Helper to toggle visibility of a widget and its label
        def set_param_visible(widget, visible):
            widget.setVisible(visible)
            label = self.blockParamsLayout.labelForField(widget)
            if label: label.setVisible(visible)

        # Toggle based on type
        set_param_visible(self.sourceFreqSpin, is_src or is_grid)
        set_param_visible(self.sourceAmpSpin, is_src or is_grid)
        set_param_visible(self.obstacleWidthSpin, is_obs)
        set_param_visible(self.obstacleHeightSpin, is_obs)
        set_param_visible(self.sourceGridNxSpin, is_grid)
        set_param_visible(self.sourceGridNySpin, is_grid)
        set_param_visible(self.sourceGridSpacingXSpin, is_grid)
        set_param_visible(self.sourceGridSpacingYSpin, is_grid)

        # Update ranges dynamically if needed (e.g., obstacle size)
        if is_obs:
            self.obstacleWidthSpin.setRange(1, self.nxSpin.value())
            self.obstacleHeightSpin.setRange(1, self.nySpin.value())
        # Could add range updates for grid size/spacing vs sim size here too

    def get_current_block_parameters(self, block_type=None):
        """Gathers parameters for the currently selected block type IN THE GUI."""
        if block_type is None: block_type = self.get_current_block_type()
        params = {}
        # Always include common source params if relevant
        if block_type == 'source' or block_type == 'source_grid':
            params.update({'frequency': self.sourceFreqSpin.value(),
                           'amplitude': self.sourceAmpSpin.value()})

        # Type-specific params
        if block_type == 'source':
            params['source_type'] = 'point'  # Explicitly set type
        elif block_type == 'obstacle':
            params.update(
                {'shape': 'rectangle', 'size': (self.obstacleWidthSpin.value(), self.obstacleHeightSpin.value()),
                 'material_type': 1})
        elif block_type == 'source_grid':
            # Include grid parameters needed for placement logic later
            params.update({
                # 'source_type': 'grid', # Not needed in final config
                'grid_nx': self.sourceGridNxSpin.value(),
                'grid_ny': self.sourceGridNySpin.value(),
                'spacing_x': self.sourceGridSpacingXSpin.value(),
                'spacing_y': self.sourceGridSpacingYSpin.value(),
            })
        return params

    @Slot(str, dict)
    def _start_canvas_drag_mode(self, item_type, params):
        # Removed check for running simulation
        self._is_dragging_config = True
        self._drag_item_type = item_type  # 'source', 'obstacle', or 'source_grid'
        self._drag_item_params = params  # Params gathered for the selected GUI type
        self.simulationView.setCursor(Qt.CrossCursor)
        # --- Use a more descriptive name for the grid ---
        display_name = item_type.replace('_', ' ').title()
        self.log_status(f"Adding {display_name}. Click on simulation area.")  # Changed prompt slightly
        self._update_drag_visual(QPointF(-100, -100))  # Place off-screen initially

    def _map_view_to_grid(self, view_pos: QPointF):
        scene_pos = self.simulationView.mapToScene(view_pos.toPoint())
        nx, ny, dx = self.nxSpin.value(), self.nySpin.value(), DEFAULT_DX
        if nx <= 0 or ny <= 0 or dx <= 0: return 0, 0  # Avoid division by zero
        # Clamp scene coordinates to valid range before converting
        scene_x = max(0.0, min(nx * dx - 1e-6, scene_pos.x()))
        scene_y = max(0.0, min(ny * dx - 1e-6, scene_pos.y()))
        # Grid coords
        grid_x = int(scene_x / dx)
        grid_y = int(((ny * dx) - scene_y) / dx)  # Invert Y
        # Final clamp to ensure indices are valid
        grid_x = max(0, min(nx - 1, grid_x))
        grid_y = max(0, min(ny - 1, grid_y))
        return grid_x, grid_y

    def _view_mouse_press(self, event):
        if self._is_dragging_config and event.button() == Qt.LeftButton:
            grid_x, grid_y = self._map_view_to_grid(event.position())
            self._finalize_drop(grid_x, grid_y)
            # Don't pass event up if we handled it
        else:
            QGraphicsView.mousePressEvent(self.simulationView, event)  # Pass event up

    def _view_mouse_move(self, event):
        if self._is_dragging_config: self._update_drag_visual(event.position())
        QGraphicsView.mouseMoveEvent(self.simulationView, event)

    def _view_mouse_release(self, event):
        # Clicking places the item, releasing doesn't do anything extra here
        # if self._is_dragging_config and event.button() == Qt.LeftButton:
        # Moved cancel to finalize/clear
        #     self._cancel_drag()
        QGraphicsView.mouseReleaseEvent(self.simulationView, event)

    def _update_drag_visual(self, view_pos: QPointF):
        """Updates the temporary visual item following the cursor."""
        if not self._is_dragging_config: return
        scene_pos = self.simulationView.mapToScene(view_pos.toPoint())
        dx = DEFAULT_DX;
        item_type = self._drag_item_type;
        params = self._drag_item_params  # Params gathered from GUI for the selected type

        # Create or update visual item
        if self._drag_visual_item is None:
            # --- Create Visual based on type ---
            if item_type == 'source':
                radius = dx * 1.5
                self._drag_visual_item = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
                self._drag_visual_item.setBrush(QColor(150, 255, 150, 180));  # Light Green
                self._drag_visual_item.setPen(Qt.NoPen)
            elif item_type == 'obstacle':
                w_grid, h_grid = params.get('size', (1, 1));
                w, h = w_grid * dx, h_grid * dx
                self._drag_visual_item = QGraphicsRectItem(-w / 2, -h / 2, w, h)  # Center on cursor
                self._drag_visual_item.setBrush(QColor(100, 100, 100, 180));  # Gray
                self._drag_visual_item.setPen(Qt.NoPen)
            elif item_type == 'source_grid':  # --- Visual for Source Grid ---
                grid_nx = params.get('grid_nx', 1);
                grid_ny = params.get('grid_ny', 1)
                spacing_x = params.get('spacing_x', 1);
                spacing_y = params.get('spacing_y', 1)
                # Calculate total extent in scene units
                w = (grid_nx - 1) * spacing_x * dx + dx  # Approx width
                h = (grid_ny - 1) * spacing_y * dx + dx  # Approx height
                # Draw a rectangle representing the grid bounds (top-left aligned with potential drop)
                self._drag_visual_item = QGraphicsRectItem(0, 0, max(dx, w), max(dx, h))  # Min size dx*dx
                self._drag_visual_item.setBrush(QColor(150, 150, 255, 100))  # Light Blue, semi-transparent
                self._drag_visual_item.setPen(QPen(QColor(50, 50, 200), 1, Qt.DashLine))

            if self._drag_visual_item:
                self._drag_visual_item.setZValue(10);
                self.simulationView.scene.addItem(self._drag_visual_item)

        # Update position (different logic for centering vs top-left)
        if self._drag_visual_item:
            if item_type == 'source' or item_type == 'obstacle':
                self._drag_visual_item.setPos(scene_pos)  # Center visual on cursor
            elif item_type == 'source_grid':
                # Map scene pos to potential top-left grid corner for the visual
                grid_x, grid_y = self._map_view_to_grid(view_pos)
                nx, ny = self.nxSpin.value(), self.nySpin.value()
                # Calculate scene coordinates for the *top-left* of the grid cell (grid_x, grid_y)
                scene_drop_x = grid_x * dx
                # Scene Y is calculated from the top, but grid_y is from bottom (inverted)
                # The top scene Y coord for grid_y is (ny - 1 - grid_y) * dx
                scene_drop_y = (ny - 1 - grid_y) * dx
                self._drag_visual_item.setPos(scene_drop_x, scene_drop_y)

    def _finalize_drop(self, grid_x, grid_y):
        """Adds the configured item(s) to the simulation configuration."""
        if not self._is_dragging_config: return

        item_type = self._drag_item_type  # The type selected in the GUI combo box
        params = self._drag_item_params  # Params gathered from the GUI for that type

        items_added_count = 0
        nx, ny = self.nxSpin.value(), self.nySpin.value()  # Get current grid size

        if item_type == 'source':
            # Add a single source item
            final_grid_pos = (grid_x, grid_y)
            item_data = {
                'type': 'source',  # Final type is 'source'
                'source_type': 'point',
                'frequency': params.get('frequency', 1.0),
                'amplitude': params.get('amplitude', 0.0),
                'position': final_grid_pos
            }
            self.placed_items_data.append(item_data)
            items_added_count = 1
            self.log_status(f"Source added at grid {final_grid_pos}.")

        elif item_type == 'obstacle':
            # Add a single obstacle item
            final_grid_pos = (grid_x, grid_y)  # Position is top-left
            item_data = {
                'type': 'obstacle',  # Final type is 'obstacle'
                'shape': 'rectangle',
                'size': params.get('size', (1, 1)),
                'material_type': params.get('material_type', 1),
                'position': final_grid_pos
            }
            self.placed_items_data.append(item_data)
            items_added_count = 1
            self.log_status(f"Obstacle added at grid {final_grid_pos} (size {params.get('size')}).")

        elif item_type == 'source_grid':
            # --- Generate multiple 'source' items ---
            grid_nx = params.get('grid_nx', 1);
            grid_ny = params.get('grid_ny', 1)
            spacing_x = params.get('spacing_x', 1);
            spacing_y = params.get('spacing_y', 1)
            frequency = params.get('frequency', 1.0)
            amplitude = params.get('amplitude', 0.0)
            base_x, base_y = grid_x, grid_y  # Use clicked point as top-left grid origin

            self.log_status(f"Generating source grid ({grid_nx}x{grid_ny}) starting at ({base_x},{base_y})...")

            for ix in range(grid_nx):
                for iy in range(grid_ny):
                    current_grid_x = base_x + ix * spacing_x
                    current_grid_y = base_y + iy * spacing_y

                    # Check bounds before adding
                    if 0 <= current_grid_x < nx and 0 <= current_grid_y < ny:
                        item_data = {
                            'type': 'source',  # <<< Use 'source' type for the simulator
                            'source_type': 'point',
                            'frequency': frequency,  # Use common frequency
                            'amplitude': amplitude,  # Use common amplitude
                            'position': (current_grid_x, current_grid_y)  # Position of this specific point
                        }
                        self.placed_items_data.append(item_data)
                        items_added_count += 1
                    # else: # Optional: Log skipped points
                    #    print(f"Skipping grid point ({current_grid_x}, {current_grid_y}) - out of bounds")

            self.log_status(f"Added {items_added_count} sources for the grid.")

        # Cleanup after drop
        self._cancel_drag();
        if items_added_count > 0:
            self.draw_configuration_overlay()  # Redraw to show the new item(s)

    def _cancel_drag(self):
        """Cleans up drag operation state and visuals."""
        if self._drag_visual_item:
            if self._drag_visual_item.scene():  # Check if it's actually in the scene
                self.simulationView.scene.removeItem(self._drag_visual_item);
            self._drag_visual_item = None
        self._is_dragging_config = False;
        self._drag_item_type = None;
        self._drag_item_params = None
        self.simulationView.setCursor(Qt.ArrowCursor)

    def draw_configuration_overlay(self):
        """Draws visual representations of placed sources and obstacles."""
        # Clear existing graphics items first
        for item in self.config_items_gfx:
            if item.scene(): self.simulationView.scene.removeItem(item)
        self.config_items_gfx.clear()

        scene = self.simulationView.scene;
        nx, ny, dx = self.nxSpin.value(), self.nySpin.value(), DEFAULT_DX
        if not scene: return  # Scene might not exist yet

        # Iterate through the stored configuration data
        for item_data in self.placed_items_data:
            item_type = item_data['type'];  # Should be 'source' or 'obstacle' now
            grid_x, grid_y = item_data.get('position', (0, 0));
            gfx_item = None

            if item_type == 'source':
                # Center ellipse on the grid cell
                scene_x, scene_y = (grid_x + 0.5) * dx, (ny - 1 - grid_y + 0.5) * dx;
                radius = dx * 1.0  # Make source points reasonably visible
                gfx_item = QGraphicsEllipseItem(scene_x - radius, scene_y - radius, radius * 2, radius * 2)
                gfx_item.setBrush(QColor("lime"));
                gfx_item.setPen(QPen(Qt.black, 0))
                gfx_item.setToolTip(
                    f"Source: ({grid_x},{grid_y})\nF:{item_data.get('frequency'):.1f} Hz\nA:{item_data.get('amplitude'):.1f}")

            elif item_type == 'obstacle':
                w_grid, h_grid = item_data.get('size', (1, 1))
                # Top-left corner for rectangle
                scene_x, scene_y = grid_x * dx, (ny - h_grid - grid_y) * dx  # Adjusted Y for top-left
                gfx_item = QGraphicsRectItem(scene_x, scene_y, w_grid * dx, h_grid * dx)
                gfx_item.setBrush(QColor(50, 50, 50));
                gfx_item.setPen(Qt.NoPen)
                gfx_item.setToolTip(f"Obstacle: ({grid_x},{grid_y})\nSize:({w_grid}x{h_grid})")

            # Add the generated graphics item
            if gfx_item:
                gfx_item.setZValue(1);  # Ensure it's above the wave pixmap
                scene.addItem(gfx_item);
                self.config_items_gfx.append(gfx_item)  # Keep track for clearing

    def clear_configuration(self):
        self.placed_items_data.clear()
        for item in self.config_items_gfx:
            if item.scene(): self.simulationView.scene.removeItem(item)
        self.config_items_gfx.clear()

        nx = self.nxSpin.value()
        ny = self.nySpin.value()
        if nx > 0 and ny > 0:
            reset_image = QImage(nx, ny, QImage.Format_RGB32)
            reset_image.fill(Qt.gray)
            reset_pixmap = QPixmap.fromImage(reset_image)
            if hasattr(self, 'wavePixmapItem') and self.wavePixmapItem:
                self.wavePixmapItem.setPixmap(reset_pixmap)
            else:  # Safety fallback if wavePixmapItem wasn't created
                self.init_scene()
        elif hasattr(self, 'wavePixmapItem') and self.wavePixmapItem:
            # If grid size is 0, remove the pixmap item
            if self.wavePixmapItem.scene():
                self.simulationView.scene.removeItem(self.wavePixmapItem)
            self.wavePixmapItem = None

        self.log_status("Configuration cleared.")
        self._cancel_drag()  # Ensure any drag operation is cancelled

    def gather_config_from_gui(self):
        """Gathers the final configuration for the simulator."""
        # Removed validation checks
        # self.placed_items_data now contains only 'source' and 'obstacle' types
        return {
            'nx': self.nxSpin.value(), 'ny': self.nySpin.value(),
            'max_time_steps': self.stepsSpin.value(), 'npml': self.pmlSpin.value(),
            'courant_num': self.courantSpin.value(), 'max_damping': self.dampingSpin.value(),
            'viz_interval': self.vizIntervalSpin.value(),
            'c': 1.0, 'dx': DEFAULT_DX,
            'building_blocks': list(self.placed_items_data)  # Pass the list of sources/obstacles
        }

    @Slot(str)
    def log_status(self, message):
        self.statusLog.append(f"[Info] {message}")

    @Slot()
    def start_simulation(self):
        if self.nxSpin.value() <= 0 or self.nySpin.value() <= 0:
            self.log_status("Invalid grid dimensions (NX, NY must be > 0).")
            return

        config = self.gather_config_from_gui()  # Gets the list of sources/obstacles
        self.statusLog.clear()
        # Re-initialize scene: important to apply potentially changed grid size
        # and ensures overlay matches the config being run
        self.init_scene()
        self.log_status("Starting simulation thread...")
        self.runButton.setEnabled(False);
        self.stopButton.setEnabled(True)
        self.set_controls_enabled(False)

        self.simulation_thread = SimulationThread(config, self)
        self.simulation_thread.frameReady.connect(self.update_simulation_display)
        self.simulation_thread.statusUpdate.connect(self.log_status)
        self.simulation_thread.simulationFinished.connect(self.on_simulation_finished)
        self.simulation_thread.start()

    @Slot()
    def stop_simulation(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.log_status("Sending stop request...")
            self.simulation_thread.stop_simulation()
            self.stopButton.setEnabled(False)  # Disable immediately on request

    @Slot(np.ndarray, float, int)
    def update_simulation_display(self, frame_data, current_time, step):
        """Updates the QPixmapItem with the latest simulation frame."""
        if not hasattr(self, 'wavePixmapItem') or not self.wavePixmapItem:
            return  # No pixmap item to update

        # Check for dimension mismatch (can happen if grid size changed before sim started fully)
        pixmap = self.wavePixmapItem.pixmap()
        # Ensure pixmap is valid before checking dimensions
        if not pixmap.isNull():
            if pixmap.width() != frame_data.shape[1] or pixmap.height() != frame_data.shape[0]:
                expected_shape = (pixmap.height(), pixmap.width())
                # Check frame_data shape is valid too
                if frame_data.ndim == 2 and all(s > 0 for s in frame_data.shape):
                    self.log_status(
                        f"Frame data shape {frame_data.shape} mismatch with pixmap {expected_shape}. Reinitializing scene.")
                    self.init_scene()  # Attempt to sync GUI state
                else:
                    self.log_status(f"Received invalid frame data shape: {frame_data.shape}. Skipping update.")
                    return  # Skip update for invalid data

                if not self.wavePixmapItem: return  # Still failed after reinit

        # --- FIX: Flip the data vertically before displaying ---
        # NumPy arrays have [0,0] at top-left, but our scene setup visually
        # places grid_y=0 at the bottom. Flip the data to match.
        flipped_frame_data = np.flipud(frame_data)
        # --- End of fix ---

        vmin, vmax = self.vminSpin.value(), self.vmaxSpin.value()

        # Use the *flipped* data for image conversion
        q_image = numpy_to_qimage(flipped_frame_data, vmin, vmax, COLORMAP)

        # Check if image conversion was successful
        if q_image.isNull():
            self.log_status("Warning: Failed to convert numpy array to QImage.")
            return

        self.wavePixmapItem.setPixmap(QPixmap.fromImage(q_image))
        # Optional: Update window title or a dedicated label
        # self.setWindowTitle(f"Sim (t={current_time:.4f}s) [Step {step}]")

    @Slot()
    def on_simulation_finished(self):
        self.log_status("Simulation thread finished.")
        self.runButton.setEnabled(True);
        # Ensure stop button is disabled only if thread truly finished/stopped
        # (It might already be disabled if stop was clicked)
        if not (self.simulation_thread and self.simulation_thread.isRunning()):
            self.stopButton.setEnabled(False)
        self.set_controls_enabled(True)
        self.simulation_thread = None  # Clear thread reference

    def set_controls_enabled(self, enabled):
        """Enable/disable GUI controls, typically during simulation run."""
        self.simSettingsGroup.setEnabled(enabled)
        self.blockSelectGroup.setEnabled(enabled)
        self.clearConfigButton.setEnabled(enabled)
        # Run/Stop buttons handled separately based on state
        self.runButton.setEnabled(enabled)
        # Stop button enabled only when running
        is_running = (self.simulation_thread is not None and self.simulation_thread.isRunning())
        self.stopButton.setEnabled(not enabled and is_running)

        if enabled: self._cancel_drag()  # Cancel any pending drag op

    def closeEvent(self, event):
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.log_status("Attempting to stop simulation on close...")
            self.stop_simulation()
            # Give the thread a moment to process the stop request
            # self.simulation_thread.wait(500) # Optional short wait
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
