"""
Name: Speaker Impedance Measurer
Author: Max Chen
Description: 
This is a GUI script written with PyQt6 to automate and speed up earphone impedance data collection with the Analog Discovery 2 from Digilent
Using the API from the DIGILENT WaveForms software, this is a custom script to directly use the existing impedance measuring function in the software.

Prerequisite:
- DIGILENT WaveForms software installed with Python API
"""

version = "0.13.2"
author = "Max Chen"

# Import necessary libraries
import os
import sys
import time
import math
import csv
import serial
import serial.tools.list_ports
from ctypes import *
from dwfconstants import *

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                           QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit,
                           QMessageBox, QWidget, QGridLayout, QGroupBox, QProgressBar,
                           QSizePolicy, QCheckBox, QTabWidget, QScrollArea,
                           QStatusBar, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# TensorFlow imports for model loading and prediction
# print("Attempting to import TensorFlow...")  # Debug line
try:
    # import tensorflow as tf
    # print(f"Successfully imported TensorFlow version: {tf.__version__}")
    # print(f"TensorFlow is using backend: {tf.config.list_physical_devices()}")
    TENSORFLOW_AVAILABLE = False  # Set to False since we're not using TensorFlow
except ImportError as e:
    print(f"Failed to import TensorFlow: {str(e)}")
    print("\nDetailed error information:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("\nPlease ensure TensorFlow is installed in your virtual environment:")
    print("1. Activate your virtual environment")
    print("2. Run: pip install tensorflow==2.19")
    print("3. If the error persists, try:")
    print("   - pip uninstall tensorflow")
    print("   - pip cache purge")
    print("   - pip install tensorflow==2.19")
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error while importing TensorFlow: {str(e)}")
    print("\nDetailed error information:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    TENSORFLOW_AVAILABLE = False

# NumPy for data processing
import numpy as np

# Matplotlib canvas class
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        plt.style.use('dark_background')
        # Use equal width and height to start with a square figure
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        self.fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        super(MplCanvas, self).__init__(self.fig)
        self.setMinimumHeight(300)
        self.setMinimumWidth(300)
        
        # Set up policy to maintain aspect ratio when resizing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

# Arduino communication thread
class ArduinoMonitor(QThread):
    data_signal = pyqtSignal(float, float, float, float, float)  # Temperature, Humidity, Pressure, Raw Sound Level, Smoothed dBA
    error_signal = pyqtSignal(str)
    
    def __init__(self, port='COM8', baud_rate=115200):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = True
        self.serial = None
        self.temperature = 25.0
        self.humidity = 50.0
        self.pressure = 1013.25  # Standard atmospheric pressure
        self.raw_sound_level = 0.0
        self.smoothed_dba = 0.0
        
    def run(self):
        try:
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=2)
            time.sleep(2)  # Allow time for Arduino to reset
            
            while self.running:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8').strip()
                        parts = line.split(',')
                        if len(parts) >= 5:  # Check for all 5 values
                            temp = float(parts[0])
                            humidity = float(parts[1])
                            pressure = float(parts[2])
                            raw_sound = float(parts[3])
                            smoothed_dba = float(parts[4])
                            
                            self.temperature = temp
                            self.humidity = humidity
                            self.pressure = pressure
                            self.raw_sound_level = raw_sound
                            self.smoothed_dba = smoothed_dba
                            
                            self.data_signal.emit(temp, humidity, pressure, raw_sound, smoothed_dba)
                    except Exception as e:
                        self.error_signal.emit(f"Error parsing Arduino data: {str(e)}")
                time.sleep(0.1)
        except Exception as e:
            self.error_signal.emit(f"Arduino connection error: {str(e)}")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()
    
    def get_latest_data(self):
        return (self.temperature, self.humidity, self.pressure, self.raw_sound_level, self.smoothed_dba)
                
    def stop(self):
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.wait()

# Worker thread for data collection
class DataCollectionWorker(QThread):
    progress_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(list, list)
    progress_bar_signal = pyqtSignal(int, int)  # Current step, total steps
    finished_signal = pyqtSignal()
    device_error_signal = pyqtSignal()  # New signal for device errors
    
    def __init__(self, app, folder_name, repetitions, record_env_data=True):
        super().__init__()
        self.app = app
        self.folder_name = folder_name
        self.repetitions = repetitions
        self.record_env_data = record_env_data
        
    def run(self):
        # Opens the device, Analog Discovery 2 through the serial port.
        hdwf = c_int()
        szerr = create_string_buffer(512)
        self.app.dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

        if hdwf.value == hdwfNone.value:
            self.app.dwf.FDwfGetLastErrorMsg(szerr)
            self.progress_signal.emit(f"Failed to open device: {str(szerr.value)}")
            self.device_error_signal.emit()  # Emit device error signal
            return
        
        # Impedance Measurement settings from the GUI
        self.app.dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3))
        sts = c_byte()
        steps = int(self.app.step_entry.text())
        start = int(self.app.start_frequency_entry.text())
        stop = int(self.app.stop_frequency_entry.text())
        reference = int(self.app.reference_entry.text())
        
        # Check if Arduino monitor is available
        arduino_available = hasattr(self.app, 'arduino_monitor') and self.app.arduino_monitor is not None

        # For loop through the specified number of runs
        for run in range(self.repetitions):
            self.progress_signal.emit(f"Starting run {run + 1} of {self.repetitions}")
            
            # Get initial environmental data for the run name/folder if available
            if arduino_available and self.record_env_data:
                temp, humidity, pressure, raw_sound, smoothed_dba = self.app.arduino_monitor.get_latest_data()
                self.progress_signal.emit(f"Initial environmental data: {temp:.1f}°C, {humidity:.1f}%, {pressure:.2f} hPa, {raw_sound:.2f} RMS, {smoothed_dba:.2f} dBA")
            else:
                # Default environmental values if no Arduino
                temp, humidity, pressure, raw_sound, smoothed_dba = 25.0, 50.0, 1013.25, 0.0, 0.0
                if self.record_env_data:  # This should actually never happen as we set record_env_data to False if no Arduino
                    self.progress_signal.emit("No environmental data available - Arduino not connected")
            
            # Configure the settings for impedance measurements
            self.app.dwf.FDwfAnalogImpedanceReset(hdwf)
            self.app.dwf.FDwfAnalogImpedanceModeSet(hdwf, c_int(0))
            self.app.dwf.FDwfAnalogImpedanceReferenceSet(hdwf, c_double(reference))
            self.app.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(start))
            self.app.dwf.FDwfAnalogImpedanceAmplitudeSet(hdwf, c_double(0.5))
            self.app.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(1))
            time.sleep(0.5)

            # Create Values for data we are interested in
            rgHz = [0.0]*steps
            rgTheta = [0.0]*steps
            rgZ = [0.0]*steps
            rgRs = [0.0]*steps
            rgXs = [0.0]*steps

            # Opens and writes to file based on the file structure
            if self.record_env_data:
                file_name = f"{self.folder_name}_Run{run + 1}_T{temp:.1f}C_H{humidity:.1f}pct.csv"
            else:
                file_name = f"{self.folder_name}_Run{run + 1}.csv"
                
            file_path = os.path.join(self.app.base_folder, file_name)
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Add environmental data to CSV header if enabled - showing initial values
                if self.record_env_data:
                    writer.writerow([f"Initial Temperature (°C): {temp:.2f}", f"Initial Humidity (%): {humidity:.2f}", f"Initial Pressure (hPa): {pressure:.2f}", f"Initial Raw Sound Level (RMS): {raw_sound:.2f}", f"Initial Smoothed dBA: {smoothed_dba:.2f}"])
                
                # Update the header row to include environmental data columns if enabled
                if self.record_env_data:
                    writer.writerow(["Frequency (Hz)", "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", "Trace Xs (Ohm)", "Temperature (°C)", "Humidity (%)", "Pressure (hPa)", "Raw Sound Level (RMS)", "Smoothed dBA"])
                else:
                    writer.writerow(["Frequency (Hz)", "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", "Trace Xs (Ohm)"])

                for i in range(steps):
                    hz = start + i * (stop - start) / (steps - 1) # linear frequency steps
                    rgHz[i] = hz
                    self.app.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(hz)) # frequency in Hertz
                    
                    # Update progress bar
                    total_steps = steps * self.repetitions
                    current_step = i + (run * steps)
                    self.progress_bar_signal.emit(current_step, total_steps)
                    
                    while True:
                        if self.app.dwf.FDwfAnalogImpedanceStatus(hdwf, byref(sts)) == 0:
                            self.app.dwf.FDwfGetLastErrorMsg(szerr)
                            print(str(szerr.value))
                            self.progress_signal.emit(f"Error: {str(szerr.value)}")
                            return
                        if sts.value == 2:
                            break
                    
                    resistance = c_double()
                    reactance = c_double()
                    self.app.dwf.FDwfAnalogImpedanceStatusMeasure(hdwf, DwfAnalogImpedanceResistance, byref(resistance))
                    self.app.dwf.FDwfAnalogImpedanceStatusMeasure(hdwf, DwfAnalogImpedanceReactance, byref(reactance))
                    rgRs[i] = resistance.value
                    rgXs[i] = reactance.value
                    rgZ[i] = (resistance.value**2 + reactance.value**2)**0.5
                    rgTheta[i] = math.degrees(math.atan2(reactance.value, resistance.value))

                    # Get the latest environmental data for each measurement point if available
                    if arduino_available and self.record_env_data:
                        current_temp, current_humidity, current_pressure, current_raw_sound, current_smoothed_dba = self.app.arduino_monitor.get_latest_data()
                        
                        # Write data to CSV with current environmental values
                        writer.writerow([hz, rgTheta[i], rgZ[i], rgRs[i], rgXs[i], 
                                       current_temp, current_humidity, current_pressure, 
                                       current_raw_sound, current_smoothed_dba])
                    else:
                        writer.writerow([hz, rgTheta[i], rgZ[i], rgRs[i], rgXs[i]])
                    
                    # Only update the text log every 10 steps to reduce UI updates
                    if i % 10 == 0 or i == steps - 1:
                        self.progress_signal.emit(f"Run {run + 1}, Step {i + 1}/{steps}: Frequency {hz:.2f} Hz, Impedance {rgZ[i]:.2f} Ohms")

            self.plot_signal.emit(rgHz, rgZ)
            time.sleep(0.5)

        # Clean up after all runs are complete
        self.app.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.app.dwf.FDwfDeviceClose(hdwf)
        self.plot_signal.emit(rgHz, rgZ)
        # self.progress_signal.emit("Data collection completed.")  # Remove this line
        self.progress_bar_signal.emit(100, 100)  # Ensure progress bar shows 100%
        self.finished_signal.emit()

# Main Data collecting class
class DataCollectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Speaker Impedance Measurement - Analog Discovery 2 (v{version})")
        
        # Set the application icon
        self.setWindowIcon(QIcon("Analog Discovery/assets/speaker_icon.png"))
        
        # Set fixed initial size
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)
        
        # Initialize prediction-related variables
        self.model = None
        self.model_path = None
        self.last_measurement = None
        self.prediction_result = None
        
        # Uncomment to auto start maximized
        self.showMaximized()
        
        # Set dark theme
        self.apply_dark_theme()
        
        # Set up the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout is horizontal - control panel on left, plot on right
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #252525;
                color: #ffffff;
                border-top: 1px solid #3d3d3d;
            }
        """)
        
        # Create progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumWidth(200)  # Limit width in status bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 1px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #58a6ff;
                width: 10px;
                margin: 0px;
            }
        """)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Arduino monitor will be initialized when the user selects a port
        self.arduino_monitor = None
        
        # Create control panel group
        self.create_control_panel()
        
        # Create plot panel
        self.create_plot_panel()
        
        # Create export dataset button
        self.export_dataset_button = QPushButton("Export ML Training Dataset")
        self.export_dataset_button.setStyleSheet("""
            QPushButton {
                background-color: #2c2c7c;
                color: white;
                font-size: 14px;
                padding: 6px;
                border-radius: 4px;
                border: none;
            }
            QPushButton[env_data="true"] {
                background-color: #2ea043;
                color: white;
            }
            QPushButton:hover {
                background-color: #3b3b9c;
            }
            QPushButton[env_data="true"]:hover {
                background-color: #3fb950;
            }
        """)
        self.export_dataset_button.clicked.connect(self.export_ml_dataset)
        self.export_dataset_button.setProperty("env_data", "false")
        self.export_dataset_button.setEnabled(True)
        self.input_layout.addWidget(self.export_dataset_button)
        
        self.statusBar().showMessage("Ready - Press Ctrl+R to start data collection")
        
        # Load DWF library
        self.load_dwf()
        
        # Initialize environmental data display with default values
        self.update_environmental_data(25.0, 50.0, 1013.25, 0.0, 0.0)
        
        # Set initial status message
        self.statusBar().showMessage("Please select a COM port and connect to Arduino")
        
        # Update the connection status text
        if hasattr(self, 'connection_status'):
            self.connection_status.setText("Not Connected")
            self.connection_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
            
        # Add keyboard shortcuts
        self.start_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)  # Changed from Ctrl+S to Ctrl+R
        self.start_shortcut.activated.connect(self.start_data_collection)
        
        # Add keyboard shortcut for saving ML dataset
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.export_ml_dataset)
            
        # Now that everything is initialized, refresh ports (and potentially auto-connect)
        QTimer.singleShot(100, self.refresh_ports)

    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                margin: 0px;
                padding: 0px;
            }
            QGroupBox {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 1ex;
                color: #ffffff;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #ffffff;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 3px;
                margin: 0px;
                max-height: 24px;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 3px;
                margin: 0px;
                max-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-width: 0px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 3px;
            }
            QPushButton {
                background-color: #2ea043;
                color: white;
                padding: 8px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3fb950;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #aaaaaa;
            }
            QProgressBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 1px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #58a6ff;
                width: 10px;
                margin: 0px;
            }
        """)

    def create_control_panel(self):
        """Create the control panel with all input widgets"""
        # Create a group box for the control panel
        self.control_group = QGroupBox("Control Panel")
        self.control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        
        # Set a size policy that allows the control panel to have a minimum width
        self.control_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.control_group.setMinimumWidth(350)
        self.control_group.setMaximumWidth(550)
        
        # Main layout for the control panel
        self.input_layout = QVBoxLayout(self.control_group)
        self.input_layout.setSpacing(5)
        self.input_layout.setContentsMargins(10, 15, 10, 10)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-bottom-color: #3d3d3d;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color: #3d3d3d;
                border-bottom-color: #3d3d3d;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)
        
        # Create the tabs
        self.main_tab = QWidget()
        self.advanced_tab = QWidget()
        self.prediction_tab = QWidget()  # Add prediction tab
        self.info_tab = QWidget()  # Add info tab
        
        # Create layouts for tabs
        self.main_tab_layout = QVBoxLayout(self.main_tab)
        self.main_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_tab_layout = QVBoxLayout(self.advanced_tab)
        self.prediction_tab_layout = QVBoxLayout(self.prediction_tab)  # Add prediction tab layout
        self.info_tab_layout = QVBoxLayout(self.info_tab)  # Add info tab layout
        
        # Add tabs to the tab widget
        self.tab_widget.addTab(self.main_tab, "Main")
        self.tab_widget.addTab(self.advanced_tab, "Advanced")
        self.tab_widget.addTab(self.prediction_tab, "Prediction")  # Add prediction tab
        self.tab_widget.addTab(self.info_tab, "Info")  # Add info tab
        
        # Add tab widget to main layout
        self.input_layout.addWidget(self.tab_widget)
        
        # Create contents for each tab
        self.create_main_tab_contents()
        self.create_advanced_tab_contents()
        self.create_prediction_tab()  # Create prediction tab contents
        self.create_info_tab()  # Create info tab contents

    def create_prediction_tab(self):
        """Create the prediction tab with model loading and prediction controls"""
        # Add a scroll area
        prediction_scroll = QScrollArea()
        prediction_scroll.setWidgetResizable(True)
        prediction_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        prediction_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        prediction_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create a container widget for the prediction tab content
        prediction_content = QWidget()
        prediction_layout = QVBoxLayout(prediction_content)
        prediction_layout.setSpacing(10)
        
        # Model Selection Section
        model_header = QLabel("Model Selection")
        model_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        prediction_layout.addWidget(model_header)
        
        # Model file selection
        model_file_layout = QHBoxLayout()
        self.model_path_label = QLabel("TensorFlow not available")
        self.model_path_label.setStyleSheet("color: #e74c3c;")
        
        select_model_button = QPushButton("Select Model")
        select_model_button.clicked.connect(self.select_model_file)
        select_model_button.setMaximumWidth(120)
        select_model_button.setEnabled(False)  # Disable since TensorFlow is not available
        
        model_file_layout.addWidget(self.model_path_label)
        model_file_layout.addWidget(select_model_button)
        prediction_layout.addLayout(model_file_layout)
        
        # Speaker Type Selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Speaker Type:")
        self.prediction_type_combo = QComboBox()
        self.prediction_type_combo.addItems(self.types)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.prediction_type_combo)
        prediction_layout.addLayout(type_layout)
        
        # Single Measurement Button
        self.measure_button = QPushButton("Take Single Measurement")
        self.measure_button.clicked.connect(self.take_single_measurement)
        self.measure_button.setStyleSheet("""
            QPushButton {
                background-color: #2ea043;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #3fb950;
            }
        """)
        self.measure_button.setEnabled(False)  # Disable since TensorFlow is not available
        prediction_layout.addWidget(self.measure_button)
        
        # Prediction Results Section
        results_header = QLabel("Prediction Results")
        results_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 15px;")
        prediction_layout.addWidget(results_header)
        
        # Results display
        self.prediction_result_label = QLabel("TensorFlow is not available. This feature is currently disabled.")
        self.prediction_result_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-height: 40px;
            }
        """)
        self.prediction_result_label.setWordWrap(True)
        prediction_layout.addWidget(self.prediction_result_label)
        
        # Add a spacer at the bottom
        prediction_layout.addStretch()
        
        # Set the scroll area widget
        prediction_scroll.setWidget(prediction_content)
        self.prediction_tab_layout.addWidget(prediction_scroll)

    def select_model_file(self):
        """Open file dialog to select a model file"""
        QMessageBox.information(self, "Feature Disabled", "TensorFlow is not available. This feature is currently disabled.")

    def take_single_measurement(self):
        """Take a single measurement for prediction"""
        QMessageBox.information(self, "Feature Disabled", "TensorFlow is not available. This feature is currently disabled.")

    def create_main_tab_contents(self):
        """Create the main tab contents"""
        # Add a scroll area to make the main tab scrollable
        self.main_scroll_area = QScrollArea()
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create a container widget for the main tab content
        self.main_content_widget = QWidget()
        self.main_content_layout = QVBoxLayout(self.main_content_widget)
        self.main_content_layout.setSpacing(10)
        
        # Start Button with improved styling
        self.start_button = QPushButton("Start Data Collection")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ea043;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                border: none;
                margin: 5px 0;
            }
            QPushButton:hover {
                background-color: #3fb950;
            }
            QPushButton:pressed {
                background-color: #238636;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #aaaaaa;
            }
        """)
        self.start_button.setToolTip("Start collecting impedance data with the current settings.\nMake sure Arduino is connected before starting.")
        self.start_button.setMinimumHeight(50)
        self.start_button.clicked.connect(self.start_data_collection)
        self.main_content_layout.addWidget(self.start_button)
        
        # Add the scroll area to the main tab layout
        self.main_tab_layout.addWidget(self.main_scroll_area)
        self.main_scroll_area.setWidget(self.main_content_widget)
        
        # Configuration section
        config_header = QLabel("Configuration")
        config_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.main_content_layout.addWidget(config_header)

        # Create a horizontal layout for the configuration elements
        config_layout = QHBoxLayout()
        config_layout.setSpacing(15)  # Add some spacing between the elements
        
        # Dropdowns and entry fields styling
        dropdown_style = """
            QComboBox { 
                padding: 3px;
                max-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """
        
        entry_style = """
            QLineEdit {
                padding: 3px;
                max-height: 24px;
                margin: 0px;
            }
        """

        # Type selection - create a container widget and layout
        type_widget = QWidget()
        type_layout = QVBoxLayout(type_widget)
        type_layout.setSpacing(3)
        type_layout.setContentsMargins(0, 0, 0, 0)
        
        self.types = ["A", "B", "C", "D"]
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.types)
        self.type_combo.setStyleSheet(dropdown_style)
        self.type_combo.setToolTip("Select the type of earphone being measured")
        
        type_label = QLabel("Earphone Type:")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        config_layout.addWidget(type_widget)

        # Length selection - create a container widget and layout
        length_widget = QWidget()
        length_layout = QVBoxLayout(length_widget)
        length_layout.setSpacing(3)
        length_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lengths = [f"{i}" for i in range(5, 31, 3)] + ["9", "24", "39", "Open", "Blocked"]
        self.length_combo = QComboBox()
        self.length_combo.addItems(self.lengths)
        self.length_combo.setStyleSheet(dropdown_style)
        self.length_combo.setToolTip("Select the length of the earphone tube in millimeters")
        
        length_label = QLabel("Length (mm):")
        length_layout.addWidget(length_label)
        length_layout.addWidget(self.length_combo)
        config_layout.addWidget(length_widget)
        
        # Repetitions - create a container widget and layout
        repetitions_widget = QWidget()
        repetitions_layout = QVBoxLayout(repetitions_widget)
        repetitions_layout.setSpacing(3)
        repetitions_layout.setContentsMargins(0, 0, 0, 0)
        
        self.repetitions_entry = QLineEdit("100")
        self.repetitions_entry.setStyleSheet(entry_style)
        self.repetitions_entry.setToolTip("Number of measurement runs to perform (each run measures the full frequency range)")
        
        repetitions_label = QLabel("Number of Repetitions:")
        repetitions_layout.addWidget(repetitions_label)
        repetitions_layout.addWidget(self.repetitions_entry)
        config_layout.addWidget(repetitions_widget)
        
        # Environmental data checkbox - create a container widget and layout
        env_checkbox_widget = QWidget()
        env_checkbox_layout = QVBoxLayout(env_checkbox_widget)
        env_checkbox_layout.setSpacing(3)
        env_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        
        self.env_data_checkbox = QCheckBox("Record Environmental Data")
        self.env_data_checkbox.setChecked(True)  # Enable by default
        self.env_data_checkbox.setStyleSheet("color: #ffffff;")
        self.env_data_checkbox.setToolTip("When enabled, real-time environmental data (temperature, humidity, pressure, sound levels)\nwill be recorded for each frequency measurement point in the CSV output files.")
        
        env_checkbox_layout.addWidget(QLabel(""))  # Empty label for alignment
        env_checkbox_layout.addWidget(self.env_data_checkbox)
        config_layout.addWidget(env_checkbox_widget)
        
        # Add the config layout to the main content layout
        self.main_content_layout.addLayout(config_layout)
        
        # Environmental readings section with improved vertical spacing
        env_header = QLabel("Environmental Readings")
        env_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 15px;")
        self.main_content_layout.addWidget(env_header)
        
        # Create a group box for environmental readings to prevent vertical squishing
        env_group = QGroupBox()
        env_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                margin-top: 0px;
                padding: 10px;
                background-color: #252525;
            }
        """)
        env_group.setMinimumHeight(200)  # Set minimum height to prevent squishing
        
        # Create a grid for environmental readings with more spacing
        env_grid = QGridLayout(env_group)
        env_grid.setSpacing(10)  # Increased spacing
        env_grid.setContentsMargins(15, 10, 15, 10)  # Add some padding
        
        # Temperature reading
        temp_label = QLabel("Temperature:")
        temp_label.setMinimumHeight(24)  # Minimum height
        self.temp_value = QLabel("25.0 °C")
        self.temp_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.temp_value.setMinimumHeight(24)  # Minimum height
        self.temp_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        env_grid.addWidget(temp_label, 0, 0)
        env_grid.addWidget(self.temp_value, 0, 1)
        
        # Humidity reading
        humidity_label = QLabel("Humidity:")
        humidity_label.setMinimumHeight(24)  # Minimum height
        self.humidity_value = QLabel("50.0 %")
        self.humidity_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.humidity_value.setMinimumHeight(24)  # Minimum height
        self.humidity_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        env_grid.addWidget(humidity_label, 1, 0)
        env_grid.addWidget(self.humidity_value, 1, 1)
        
        # Pressure reading
        pressure_label = QLabel("Pressure:")
        pressure_label.setMinimumHeight(24)  # Minimum height
        self.pressure_value = QLabel("1013.25 hPa")
        self.pressure_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.pressure_value.setMinimumHeight(24)  # Minimum height
        self.pressure_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        env_grid.addWidget(pressure_label, 2, 0)
        env_grid.addWidget(self.pressure_value, 2, 1)
        
        # Raw Sound Level reading
        raw_sound_label = QLabel("Raw Sound:")
        raw_sound_label.setMinimumHeight(24)  # Minimum height
        self.raw_sound_value = QLabel("0.0 RMS")
        self.raw_sound_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.raw_sound_value.setMinimumHeight(24)  # Minimum height
        self.raw_sound_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        env_grid.addWidget(raw_sound_label, 3, 0)
        env_grid.addWidget(self.raw_sound_value, 3, 1)
        
        # Smoothed dBA reading
        smoothed_dba_label = QLabel("Smoothed dBA:")
        smoothed_dba_label.setMinimumHeight(24)  # Minimum height
        self.smoothed_dba_value = QLabel("0.0 dBA")
        self.smoothed_dba_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        self.smoothed_dba_value.setMinimumHeight(24)  # Minimum height
        self.smoothed_dba_value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        env_grid.addWidget(smoothed_dba_label, 4, 0)
        env_grid.addWidget(self.smoothed_dba_value, 4, 1)
        
        self.main_content_layout.addWidget(env_group)
        
        # Add a spacer to push everything to the top of the main tab
        self.main_content_layout.addStretch()
        
    def create_advanced_tab_contents(self):
        """Create the advanced tab contents"""
        # Define styles
        entry_style = """
            QLineEdit {
                padding: 3px;
                max-height: 24px;
                margin: 0px;
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
        """
        
        # Arduino Connection Section
        arduino_header = QLabel("Arduino Connection")
        arduino_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.advanced_tab_layout.addWidget(arduino_header)
        
        # COM Port selection
        self.port_layout = QHBoxLayout()
        port_label = QLabel("COM Port:")
        self.port_combo = QComboBox()
        self.port_combo.setToolTip("Select the COM port where your Arduino is connected")
        self.refresh_port_button = QPushButton("↻")
        self.refresh_port_button.setToolTip("Refresh the list of available COM ports")
        self.refresh_port_button.setMaximumWidth(30)
        self.refresh_port_button.clicked.connect(self.refresh_ports)
        
        # Don't call refresh_ports here - we'll do it after full initialization
        
        self.port_layout.addWidget(port_label)
        self.port_layout.addWidget(self.port_combo, 1)  # Give it more space
        self.port_layout.addWidget(self.refresh_port_button)
        
        self.reconnect_button = QPushButton("Connect")
        self.reconnect_button.setToolTip("Connect to the Arduino on the selected COM port")
        self.reconnect_button.clicked.connect(self.reconnect_arduino)
        
        self.advanced_tab_layout.addLayout(self.port_layout)
        self.advanced_tab_layout.addWidget(self.reconnect_button)
        
        # Connection status label
        self.connection_status = QLabel("Not Connected")
        self.connection_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.connection_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_status.setMinimumHeight(25)
        self.advanced_tab_layout.addWidget(self.connection_status)
        
        # Measurement Settings section
        measurement_header = QLabel("Measurement Settings")
        measurement_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 15px;")
        self.advanced_tab_layout.addWidget(measurement_header)
        
        # Create horizontal layouts for paired settings
        freq_layout = QHBoxLayout()
        freq_layout.setSpacing(10)
        
        # Start Frequency
        start_freq_widget = QWidget()
        start_freq_layout = QVBoxLayout(start_freq_widget)
        start_freq_layout.setSpacing(3)
        start_freq_layout.setContentsMargins(0, 0, 0, 0)
        self.start_frequency_entry = QLineEdit("20")
        self.start_frequency_entry.setStyleSheet(entry_style)
        self.start_frequency_entry.setToolTip("Starting frequency for the impedance sweep (Hz)")
        start_freq_layout.addWidget(QLabel("Start Frequency (Hz):"))
        start_freq_layout.addWidget(self.start_frequency_entry)
        freq_layout.addWidget(start_freq_widget)
        
        # Stop Frequency
        stop_freq_widget = QWidget()
        stop_freq_layout = QVBoxLayout(stop_freq_widget)
        stop_freq_layout.setSpacing(3)
        stop_freq_layout.setContentsMargins(0, 0, 0, 0)
        self.stop_frequency_entry = QLineEdit("20000")
        self.stop_frequency_entry.setStyleSheet(entry_style)
        self.stop_frequency_entry.setToolTip("Ending frequency for the impedance sweep (Hz)")
        stop_freq_layout.addWidget(QLabel("Stop Frequency (Hz):"))
        stop_freq_layout.addWidget(self.stop_frequency_entry)
        freq_layout.addWidget(stop_freq_widget)
        
        self.advanced_tab_layout.addLayout(freq_layout)
        
        # Reference and Steps in another horizontal layout
        ref_steps_layout = QHBoxLayout()
        ref_steps_layout.setSpacing(10)
        
        # Reference
        ref_widget = QWidget()
        ref_layout = QVBoxLayout(ref_widget)
        ref_layout.setSpacing(3)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_entry = QLineEdit("100")
        self.reference_entry.setStyleSheet(entry_style)
        self.reference_entry.setToolTip("Reference resistance value in Ohms for impedance measurement")
        ref_layout.addWidget(QLabel("Reference in Ohms:"))
        ref_layout.addWidget(self.reference_entry)
        ref_steps_layout.addWidget(ref_widget)
        
        # Steps
        steps_widget = QWidget()
        steps_layout = QVBoxLayout(steps_widget)
        steps_layout.setSpacing(3)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        self.step_entry = QLineEdit("501")
        self.step_entry.setStyleSheet(entry_style)
        self.step_entry.setToolTip("Number of frequency points to measure between start and stop frequencies")
        steps_layout.addWidget(QLabel("STEPS:"))
        steps_layout.addWidget(self.step_entry)
        ref_steps_layout.addWidget(steps_widget)
        
        self.advanced_tab_layout.addLayout(ref_steps_layout)
        
        # Add a spacer to the advanced tab to push everything to the top
        self.advanced_tab_layout.addStretch()
        
        # Status section below the tabs
        status_header = QLabel("Status")
        status_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 15px;")
        self.input_layout.addWidget(status_header)

        # Progress text area
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMinimumHeight(60)
        self.progress_text.setMaximumHeight(100)
        self.progress_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        self.input_layout.addWidget(self.progress_text)

        # Add control group to the main layout
        self.main_layout.addWidget(self.control_group, stretch=1)

    def create_plot_panel(self):
        """Create the plot panel with matplotlib canvas"""
        # Create a group box for the plot
        self.plot_group = QGroupBox("Impedance Plot")
        self.plot_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        
        # Set size policy to allow the plot to expand and maintain aspect ratio
        self.plot_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_group.setMinimumWidth(350)
        
        plot_layout = QVBoxLayout(self.plot_group)
        plot_layout.setSpacing(5)
        plot_layout.setContentsMargins(10, 15, 10, 10)

        # Create the matplotlib canvas with dark theme - using square dimensions
        self.canvas = MplCanvas(self, width=4, height=4, dpi=100)
        self.canvas.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.canvas.ax.set_xlabel('Frequency (Hz)')
        self.canvas.ax.set_ylabel('Impedance (Ohms)')
        self.canvas.ax.set_xscale('log')
        self.canvas.ax.set_yscale('log')
        
        plot_layout.addWidget(self.canvas)
        
        # Add plot group to the main layout - with reduced stretch factor
        self.main_layout.addWidget(self.plot_group, stretch=1)

    def update_environmental_data(self, temperature, humidity, pressure, raw_sound, smoothed_dba):
        """Update the environmental data labels with values from Arduino"""
        self.temp_value.setText(f"{temperature:.1f} °C")
        self.humidity_value.setText(f"{humidity:.1f} %")
        self.pressure_value.setText(f"{pressure:.2f} hPa")
        self.raw_sound_value.setText(f"{raw_sound:.2f} RMS")
        self.smoothed_dba_value.setText(f"{smoothed_dba:.2f} dBA")
        
    def handle_arduino_error(self, error_message):
        """Handle Arduino connection errors"""
        self.update_progress(f"Arduino Error: {error_message}")
        # Update the environmental labels to show error state
        self.temp_value.setText("Error")
        self.humidity_value.setText("Error")
        self.pressure_value.setText("Error")
        self.raw_sound_value.setText("Error")
        self.smoothed_dba_value.setText("Error")
        self.temp_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.humidity_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.pressure_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.raw_sound_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.smoothed_dba_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        
        # Update connection status
        self.connection_status.setText("Connection Failed")
        self.connection_status.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def load_dwf(self):
        """Load the appropriate DWF library based on platform"""
        if sys.platform.startswith("win"):
            self.dwf = cdll.LoadLibrary("dwf.dll")
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")
        self.update_progress("System initialized and ready to collect data.")

    def update_progress(self, message):
        """Update the progress text with timestamped messages"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.progress_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.progress_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update the status bar message
        self.statusBar().showMessage(message)

    def update_progress_bar(self, current, total):
        """Update the progress bar"""
        progress_percentage = min(100, int((current / total) * 100)) if total > 0 else 0
        self.progress_bar.setValue(progress_percentage)

    def update_plot(self, frequencies, impedance):
        """Update the plot with new impedance data"""
        self.canvas.ax.clear()
        
        # Ensure valid data for log scaling
        valid_frequencies = [f for f in frequencies if f > 0]
        valid_impedances = [z for z in impedance if z > 0]

        if valid_frequencies and valid_impedances:
            self.canvas.ax.plot(valid_frequencies, valid_impedances, label='|Z| (Ohms)', 
                              color='#58a6ff', linewidth=2)
            self.canvas.ax.set_xscale('log')
            self.canvas.ax.set_yscale('log')
            
            # Set axis limits to ensure data is properly visible
            x_min, x_max = min(valid_frequencies), max(valid_frequencies)
            y_min, y_max = min(valid_impedances), max(valid_impedances)
            
            # Add some padding to the limits
            x_padding = (x_max / x_min) ** 0.1
            y_padding = (y_max / y_min) ** 0.1
            
            self.canvas.ax.set_xlim(x_min / x_padding, x_max * x_padding)
            self.canvas.ax.set_ylim(y_min / y_padding, y_max * y_padding)
        else:
            self.canvas.ax.plot(frequencies, impedance, label='|Z| (Ohms)',
                              color='#58a6ff', linewidth=2)
        
        self.canvas.ax.set_xlabel('Frequency (Hz)')
        self.canvas.ax.set_ylabel('Impedance (Ohms)')
        self.canvas.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.canvas.ax.grid(True, linestyle='--', alpha=0.7, color='#333333')
        self.canvas.ax.legend(loc='best', frameon=True)
        
        # Set square aspect ratio for the plot
        self.canvas.ax.set_box_aspect(1.0)
        
        # Update the canvas
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def start_data_collection(self):
        """Start the data collection process in a separate thread"""
        # Check if Arduino is connected - now shows a warning instead of preventing measurements
        arduino_connected = self.arduino_monitor is not None
        
        if not arduino_connected:
            # Create a custom dialog with styled buttons
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Arduino Not Connected")
            msg_box.setText("Arduino is not connected. Environmental data will not be recorded.\n\nDo you want to continue without environmental data?")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            
            # Create custom styled buttons
            yes_button = msg_box.addButton('Yes', QMessageBox.ButtonRole.YesRole)
            no_button = msg_box.addButton('No', QMessageBox.ButtonRole.NoRole)
            
            # Style the buttons
            yes_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ea043;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                    border: none;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #3fb950;
                }
            """)
            
            no_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                    border: none;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            
            msg_box.setDefaultButton(no_button)
            msg_box.exec()
            
            if msg_box.clickedButton() == no_button:
                self.statusBar().showMessage("Measurement cancelled - Connect Arduino for environmental data")
                return
            
        # Get values from input fields
        selected_type = self.type_combo.currentText()
        selected_length = self.length_combo.currentText()
        
        try:
            repetitions = int(self.repetitions_entry.text())
            
            if repetitions <= 0:
                raise ValueError("Number of repetitions must be a positive integer.")
                
            start_freq = int(self.start_frequency_entry.text())
            stop_freq = int(self.stop_frequency_entry.text())
            steps = int(self.step_entry.text())
            reference = int(self.reference_entry.text())
            
            if start_freq <= 0 or stop_freq <= 0 or start_freq >= stop_freq:
                raise ValueError("Frequency range must be valid (start < stop and both > 0).")
                
            if steps <= 1:
                raise ValueError("Number of steps must be greater than 1.")
                
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            self.statusBar().showMessage(f"Error: {str(e)}")
            return
        
        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        
        # If environmental data recording is enabled and Arduino is connected, append temp/humidity to folder name
        if self.env_data_checkbox.isChecked() and arduino_connected:
            # Get current environmental data
            temp, humidity, pressure, raw_sound, smoothed_dba = self.arduino_monitor.get_latest_data()
            folder_name = f"{folder_name}_T{temp:.1f}C_H{humidity:.1f}pct"
            
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
            self.update_progress(f"Created data folder: {folder_name}")
        
        # Update UI
        self.start_button.setText("Running...")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f0883e;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                border: none;
                margin: 5px 0;
            }
        """)
        self.start_button.setEnabled(False)
        
        # Update status bar
        status_msg = f"Running {repetitions} measurements for Type {selected_type}, Length {selected_length}"
        if not arduino_connected:
            status_msg += " (without environmental data)"
        status_msg += " - Please wait..."
        self.statusBar().showMessage(status_msg)
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Determine if we should record environmental data (needs Arduino connected and checkbox enabled)
        record_env_data = self.env_data_checkbox.isChecked() and arduino_connected
        
        # Start data collection in a worker thread
        self.worker = DataCollectionWorker(self, folder_name, repetitions, record_env_data)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.plot_signal.connect(self.update_plot)
        self.worker.progress_bar_signal.connect(self.update_progress_bar)
        self.worker.finished_signal.connect(self.collection_finished)
        self.worker.device_error_signal.connect(self.handle_device_error)  # Connect device error signal
        self.worker.start()
        
        env_data_status = "with" if record_env_data else "without"
        self.update_progress(f"Starting data collection for {selected_type}_{selected_length} {env_data_status} environmental data, {repetitions} repetitions")

    def collection_finished(self):
        """Update UI after data collection is complete"""
        self.start_button.setText("Start Data Collection")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ea043;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                border: none;
                margin: 5px 0;
            }
            QPushButton:hover {
                background-color: #3fb950;
            }
            QPushButton:pressed {
                background-color: #238636;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #aaaaaa;
            }
        """)
        self.start_button.setEnabled(True)
        
        # Update status bar
        self.statusBar().showMessage("Data collection complete - Ready for next measurement")
        
        # Enable/disable export button based on environmental data availability
        arduino_was_connected = hasattr(self, 'arduino_monitor') and self.arduino_monitor is not None
        has_env_data = hasattr(self, 'worker') and self.worker.record_env_data and arduino_was_connected
        
        # Update export button state
        self.export_dataset_button.setProperty("env_data", "true" if has_env_data else "false")
        
        # Show completion message
        message = "All measurement runs have been completed successfully!\n\nThe data has been saved to CSV files in the specified folder."
        
        # Add note about environmental data if needed
        if not arduino_was_connected:
            message += "\n\nNote: No environmental data was recorded as Arduino was not connected."
            
        QMessageBox.information(self, "Data Collection Complete", message)
        
        # Play a sound to notify the user
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.MessageBeep()
        except:
            pass  # Fallback silently if sound cannot be played
        
        self.update_progress("Data collection completed.")

    def export_ml_dataset(self):
        """Export a consolidated dataset for machine learning training"""
        try:
            # Create a folder for ML datasets if it doesn't exist
            ml_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML_Datasets")
            if not os.path.exists(ml_folder):
                os.makedirs(ml_folder)
                
            # Extract type and length from the folder name
            folder_parts = os.path.basename(self.base_folder).split('_')
            if len(folder_parts) >= 2:
                earphone_type = folder_parts[0]
                earphone_length = folder_parts[1]
                
                # Check if there's environmental data in the folder name
                temperature_str = "N/A"
                humidity_str = "N/A"
                
                if len(folder_parts) >= 4 and folder_parts[2].startswith('T') and folder_parts[3].startswith('H'):
                    # Extract temperature and humidity from folder name
                    temperature_str = folder_parts[2][1:-1]  # Remove 'T' prefix and 'C' suffix
                    humidity_str = folder_parts[3][1:-3]     # Remove 'H' prefix and 'pct' suffix
            else:
                earphone_type = "unknown"
                earphone_length = "unknown"
                
            # Create the output file name based on type and length
            has_env_data = self.export_dataset_button.property("env_data") == "true"
            if has_env_data:
                output_file = os.path.join(ml_folder, f"{earphone_type}_{earphone_length}_T{temperature_str}_H{humidity_str}_All.csv")
            else:
                output_file = os.path.join(ml_folder, f"{earphone_type}_{earphone_length}_NoEnv_All.csv")
            
            # Check if file already exists
            file_exists = os.path.exists(output_file)
            
            # Get all CSV files in the current measurement folder
            csv_files = [f for f in os.listdir(self.base_folder) if f.endswith('.csv') and ('Run' in f)]
            
            # Sort the files by run number
            def get_run_number(filename):
                # Extract the run number from the filename
                parts = filename.split('_')
                for part in parts:
                    if part.startswith('Run'):
                        return int(part[3:].split('.')[0].split('_')[0])
                return 0
                
            csv_files.sort(key=get_run_number)
            
            if not csv_files:
                self.update_progress("No data files found to export.")
                return
            
            # If file exists, confirm overwrite
            if file_exists:
                reply = QMessageBox.question(
                    self, 
                    'File Exists', 
                    f'The file "{os.path.basename(output_file)}" already exists.\nDo you want to overwrite it?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    self.update_progress("ML dataset export cancelled.")
                    return
            
            # Write header and data to the consolidated file
            with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                
                # Write appropriate header based on whether we have environmental data
                if has_env_data:
                    writer.writerow([
                        "Type", "Length", "Run", "Frequency (Hz)", 
                        "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", 
                        "Trace Xs (Ohm)", "Temperature (°C)", "Humidity (%)",
                        "Pressure (hPa)", "Raw Sound Level (RMS)", "Smoothed dBA"
                    ])
                else:
                    writer.writerow([
                        "Type", "Length", "Run", "Frequency (Hz)", 
                        "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", 
                        "Trace Xs (Ohm)"
                    ])
                
                # Process each CSV file
                for i, file_name in enumerate(csv_files):
                    run_number = i + 1
                    file_path = os.path.join(self.base_folder, file_name)
                    
                    with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
                        reader = csv.reader(infile)
                        
                        # Skip header rows (2 or 3 rows depending on env data)
                        rows = list(reader)
                        header_rows = 2 if "Temperature" in rows[0][0] else 1
                        
                        # Check if this file has environmental data
                        has_file_env_data = len(rows[header_rows]) > 5
                        
                        # Process data rows
                        for row in rows[header_rows:]:
                            if not row:  # Skip empty rows
                                continue
                                
                            # Extract data
                            if has_file_env_data and has_env_data:
                                # Now each row has its own environmental measurements
                                freq, theta, z, rs, xs, temp, humidity, pressure, raw_sound, smoothed_dba = row
                                # Write consolidated row with environmental data
                                writer.writerow([
                                    earphone_type, earphone_length, run_number,
                                    freq, theta, z, rs, xs, temp, humidity,
                                    pressure, raw_sound, smoothed_dba
                                ])
                            else:
                                # Write consolidated row without environmental data
                                freq, theta, z, rs, xs = row[:5]  # Only take first 5 columns
                                writer.writerow([
                                    earphone_type, earphone_length, run_number,
                                    freq, theta, z, rs, xs
                                ])
            
            self.update_progress(f"ML training dataset exported to: {os.path.basename(output_file)}")
            
            # Ask if user wants to open the folder
            reply = QMessageBox.question(
                self, 
                'Dataset Exported', 
                f'Dataset was exported as:\n{os.path.basename(output_file)}\n\nWould you like to open the folder?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Open the folder in file explorer
                if sys.platform.startswith('win'):
                    os.startfile(ml_folder)
                elif sys.platform.startswith('darwin'):  # macOS
                    import subprocess
                    subprocess.Popen(['open', ml_folder])
                else:  # Linux
                    import subprocess
                    subprocess.Popen(['xdg-open', ml_folder])
                    
        except Exception as e:
            self.update_progress(f"Error exporting dataset: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export dataset: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop Arduino monitoring thread before closing
        if hasattr(self, 'arduino_monitor') and self.arduino_monitor is not None:
            self.arduino_monitor.stop()
            
        if hasattr(self, 'worker') and self.worker.isRunning():
            # Create message box with styled buttons
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('Exit')
            msg_box.setText('Data collection is still in progress. Are you sure you want to exit?')
            msg_box.setIcon(QMessageBox.Icon.Warning)
            
            # Create custom styled buttons
            yes_button = msg_box.addButton('Yes', QMessageBox.ButtonRole.YesRole)
            no_button = msg_box.addButton('No', QMessageBox.ButtonRole.NoRole)
            
            # Style the buttons
            yes_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                    border: none;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            
            no_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ea043;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                    border: none;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #3fb950;
                }
            """)
            
            msg_box.setDefaultButton(no_button)
            msg_box.exec()
            
            if msg_box.clickedButton() == yes_button:
                self.worker.terminate()  # Force terminate the worker thread
                event.accept()
            else:
                event.ignore()
        else:
            # Create message box with styled buttons
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('Exit')
            msg_box.setText('Are you sure you want to exit the application?')
            msg_box.setIcon(QMessageBox.Icon.Question)
            
            # Create custom styled buttons
            yes_button = msg_box.addButton('Yes', QMessageBox.ButtonRole.YesRole)
            no_button = msg_box.addButton('No', QMessageBox.ButtonRole.NoRole)
            
            # Style the buttons
            yes_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                    border: none;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            
            no_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ea043;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                    border: none;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #3fb950;
                }
            """)
            
            msg_box.setDefaultButton(no_button)
            msg_box.exec()
            
            if msg_box.clickedButton() == yes_button:
                event.accept()
            else:
                event.ignore()

    def refresh_ports(self):
        """Refresh the available COM ports list"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
            available_ports.append(port.device)
        
        # If no ports found, add a message
        if self.port_combo.count() == 0:
            self.port_combo.addItem("No COM ports found")
            self.connection_status.setText("No devices available")
            self.connection_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        # Auto-connect if only one port is detected
        elif len(available_ports) == 1:
            # Select the only available port
            self.port_combo.setCurrentIndex(0)
            # Auto-connect to it
            self.update_progress(f"Auto-connecting to the only available port: {available_ports[0]}")
            self.reconnect_arduino()
            
    def reconnect_arduino(self):
        """Reconnect to Arduino with selected COM port"""
        # Check if we have ports available
        if self.port_combo.currentText() == "No COM ports found":
            self.update_progress("Error: No COM ports available")
            self.connection_status.setText("No devices available")
            self.connection_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
            
        # Extract the COM port from the selected item
        port = self.port_combo.currentText().split(" - ")[0]
        
        # Stop current monitor if it exists and is running
        if hasattr(self, 'arduino_monitor') and self.arduino_monitor is not None:
            self.arduino_monitor.stop()
            
        # Update progress
        self.update_progress(f"Connecting to Arduino on {port}...")
        self.connection_status.setText(f"Connecting to {port}...")
        self.connection_status.setStyleSheet("color: #f39c12; font-weight: bold;") # Orange for connecting
        
        # Create new monitor
        self.arduino_monitor = ArduinoMonitor(port=port)
        self.arduino_monitor.data_signal.connect(self.update_environmental_data)
        self.arduino_monitor.error_signal.connect(self.handle_arduino_error)
        self.arduino_monitor.start()
        
        # Update button text temporarily
        self.reconnect_button.setText("Connecting...")
        self.reconnect_button.setEnabled(False)
        
        # Re-enable after short delay
        def enable_button():
            self.reconnect_button.setText("Connect")
            self.reconnect_button.setEnabled(True)
            self.connection_status.setText(f"Connected to {port}")
            self.connection_status.setStyleSheet("color: #2ecc71; font-weight: bold;") # Green for connected
            self.update_progress(f"Successfully connected to Arduino on {port}")
            self.statusBar().showMessage("Ready - Press Ctrl+R to start data collection")
            
        QTimer.singleShot(2000, enable_button)

    def create_info_tab(self):
        """Create the info tab with application information and usage instructions"""
        # Add a scroll area
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create a container widget for the info tab content
        info_content = QWidget()
        info_layout = QVBoxLayout(info_content)
        info_layout.setSpacing(15)
        
        # Application Title
        title = QLabel("Speaker Impedance Measurement Tool")
        title.setStyleSheet("""
            QLabel {
                color: #58a6ff;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        info_layout.addWidget(title)
        
        # Version and Author
        version_info = QLabel(f"Version: {version}\nAuthor: {author}")
        version_info.setStyleSheet("font-size: 14px; color: #8b949e;")
        info_layout.addWidget(version_info)
        
        # Description Section
        description_header = QLabel("Description")
        description_header.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff; margin-top: 20px;")
        info_layout.addWidget(description_header)
        
        description_text = QLabel(
            "This application automates and speeds up earphone impedance data collection "
            "using the Analog Discovery 2 from Digilent. It utilizes the API from the "
            "DIGILENT WaveForms software to directly access the impedance measuring function."
        )
        description_text.setWordWrap(True)
        description_text.setStyleSheet("font-size: 14px; line-height: 1.4;")
        info_layout.addWidget(description_text)
        
        # Prerequisites Section
        prereq_header = QLabel("Prerequisites")
        prereq_header.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff; margin-top: 20px;")
        info_layout.addWidget(prereq_header)
        
        prereq_text = QLabel(
            "• DIGILENT WaveForms software installed with Python API\n"
            "• Analog Discovery 2 hardware connected\n"
            "• Arduino (optional) for environmental data collection"
        )
        prereq_text.setStyleSheet("font-size: 14px; line-height: 1.4;")
        info_layout.addWidget(prereq_text)
        
        # Usage Instructions Section
        usage_header = QLabel("How to Use")
        usage_header.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff; margin-top: 20px;")
        info_layout.addWidget(usage_header)
        
        usage_text = QLabel(
            "1. Main Settings Tab:\n"
            "   • Select earphone type and length\n"
            "   • Set number of measurement repetitions\n"
            "   • Monitor environmental data (if Arduino connected)\n\n"
            "2. Advanced Settings Tab:\n"
            "   • Configure Arduino connection\n"
            "   • Set frequency sweep parameters\n"
            "   • Adjust reference resistance\n\n"
            "3. Prediction Tab:\n"
            "   • Load a trained model\n"
            "   • Take single measurements\n"
            "   • Get predictions for tube length\n\n"
            "4. Data Collection:\n"
            "   • Press 'Start Data Collection' or use Ctrl+R\n"
            "   • Data is saved in CSV format\n"
            "   • Real-time plot updates during measurement"
        )
        usage_text.setStyleSheet("font-size: 14px; line-height: 1.4;")
        usage_text.setWordWrap(True)
        info_layout.addWidget(usage_text)
        
        # Keyboard Shortcuts Section
        shortcuts_header = QLabel("Keyboard Shortcuts")
        shortcuts_header.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff; margin-top: 20px;")
        info_layout.addWidget(shortcuts_header)
        
        shortcuts_text = QLabel("Ctrl+R: Start Data Collection\nCtrl+S: Save ML Dataset")
        shortcuts_text.setStyleSheet("font-size: 14px; line-height: 1.4;")
        info_layout.addWidget(shortcuts_text)
        
        # Add a spacer at the bottom
        info_layout.addStretch()
        
        # Set the scroll area widget
        info_scroll.setWidget(info_content)
        self.info_tab_layout.addWidget(info_scroll)

    def handle_device_error(self):
        """Handle device connection errors"""
        # Set the start button to stopped state
        self.start_button.setText("Stopped")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                border: none;
                margin: 5px 0;
            }
        """)
        self.start_button.setEnabled(True)
        
        # Set the measure button to stopped state if it exists
        if hasattr(self, 'measure_button'):
            self.measure_button.setText("Stopped")
            self.measure_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                    margin: 10px 0;
                }
            """)
            self.measure_button.setEnabled(True)
            
        self.statusBar().showMessage("Device connection error - Please check connection")
        
        # Create a timer to reset the start button after 5 seconds
        def reset_start_button():
            self.start_button.setText("Start Data Collection")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ea043;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                    border: none;
                    margin: 5px 0;
                }
                QPushButton:hover {
                    background-color: #3fb950;
                }
                QPushButton:pressed {
                    background-color: #238636;
                }
                QPushButton:disabled {
                    background-color: #444444;
                    color: #aaaaaa;
                }
            """)
            
        # Create a timer to reset the measure button after 5 seconds
        def reset_measure_button():
            if hasattr(self, 'measure_button'):
                self.measure_button.setText("Take Single Measurement")
                self.measure_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2ea043;
                        color: white;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 6px;
                        margin: 10px 0;
                    }
                    QPushButton:hover {
                        background-color: #3fb950;
                    }
                """)
        
        # Set up the timers
        QTimer.singleShot(5000, reset_start_button)
        QTimer.singleShot(5000, reset_measure_button)

# Main function that starts the program
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = DataCollectionApp()
    window.show()
    
    sys.exit(app.exec()) 