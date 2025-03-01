"""
Name: Impedance Measurement GUI
Author: Max Chen
Date: March 1, 2024
Description: 
This is a GUI application for impedance measurement using the Analog Discovery 2.
This version is intended for predicting earphone length from a measurement taken.
"""

# Import necessary libraries
import os
import sys
import time
import csv
import threading
import numpy as np
from ctypes import *
from PyQt6 import QtWidgets, QtCore, QtGui
import serial

# Add parent directory to path to import dwfconstants
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "Analog Discovery"))
from dwfconstants import *

class ImpedancePredictionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earphones Impedance Measurement")
        self.setGeometry(100, 100, 800, 500)
        
        # Create and set the application icon
        icon_path = os.path.join(current_dir, "assets", "speaker_icon.png")
        if not os.path.exists(os.path.dirname(icon_path)):
            os.makedirs(os.path.dirname(icon_path))
            
        # Create a simple speaker icon if it doesn't exist
        if not os.path.exists(icon_path):
            self.create_speaker_icon(icon_path)
            
        app_icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(app_icon)
        QtWidgets.QApplication.setWindowIcon(app_icon)
        
        # Set dark background for the entire application
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
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
                padding: 3px;
                margin: 0px;
            }
        """)

        # Main layout (vertical)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setSpacing(5)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create a horizontal layout for the main content
        self.content_layout = QtWidgets.QHBoxLayout()
        self.content_layout.setSpacing(5)
        
        # Progress text box at the bottom
        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(80)  # Reduce height
        self.main_layout.addWidget(QtWidgets.QLabel("Status:"))
        self.main_layout.addWidget(self.progress_text)
        
        # Initialize serial communication for Arduino (temperature, humidity, pressure)
        self.serial_port = None
        try:
            self.serial_port = serial.Serial('COM8', 115200, timeout=1)  # Adjust COM port as necessary
            self.update_progress("Arduino connected.")
        except serial.SerialException:
            self.update_progress("Warning: Arduino not connected. Data collection will not include temperature, humidity, and pressure.")
        
        # Create left panel for measurement controls
        self.measurement_panel = QtWidgets.QVBoxLayout()
        self.setup_measurement_panel()
        
        # Load DWF library
        self.load_dwf()
        
        # Initialize data storage
        self.measurement_data = None
        self.frequencies = None
        self.impedance = None
        self.resistance = None
        self.reactance = None
        
        # Timer for reading Arduino data
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.read_arduino_data)
        
        # Create the prediction panel (UI only, no functionality)
        self.prediction_panel = QtWidgets.QVBoxLayout()
        self.setup_prediction_panel()
        
        # Add panels to content layout
        self.content_layout.addLayout(self.measurement_panel)
        self.content_layout.addLayout(self.prediction_panel)
        
        # Add content layout to main layout - MOVE THIS AFTER creating the progress_text
        self.main_layout.insertLayout(0, self.content_layout)

    def setup_measurement_panel(self):
        # Create a group box for measurement settings
        self.measurement_group = QtWidgets.QGroupBox("Measurement Settings")
        self.measurement_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        self.measurement_layout = QtWidgets.QVBoxLayout(self.measurement_group)
        self.measurement_layout.setSpacing(5)  # Reduce spacing
        self.measurement_layout.setContentsMargins(5, 8, 5, 5)  # Reduce margins
        
        # Create sections with headers
        # Section 1: Measurement Control
        control_section = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_section)
        control_layout.setSpacing(5)
        
        # Start Button with improved styling for dark theme
        self.start_button = QtWidgets.QPushButton("Start Measurement")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ea043;
                color: white;
                font-size: 18px;
                padding: 8px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3fb950;
            }
        """)
        self.start_button.clicked.connect(self.start_measurement)
        control_layout.addWidget(self.start_button)
        
        # Section 2: Configuration
        config_section = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_section)
        config_layout.setSpacing(5)
        
        # Add section header
        config_header = QtWidgets.QLabel("Configuration")
        config_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #58a6ff;")
        config_layout.addWidget(config_header)
        
        # Dropdowns with improved styling
        dropdown_style = "QComboBox { padding: 5px; } QComboBox::drop-down { border: none; }"
        
        self.types = ["A", "B", "C", "D"]
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(self.types)
        self.type_combo.setStyleSheet(dropdown_style)
        config_layout.addWidget(QtWidgets.QLabel("Earphone Type:"))
        config_layout.addWidget(self.type_combo)
        
        self.lengths = [str(i) for i in range(5, 31, 3)] + [str(i) for i in [9, 24, 39]] + ["Open", "Blocked"]
        self.length_combo = QtWidgets.QComboBox()
        self.length_combo.addItems(self.lengths)
        self.length_combo.setStyleSheet(dropdown_style)
        config_layout.addWidget(QtWidgets.QLabel("Length (mm):"))
        config_layout.addWidget(self.length_combo)
        
        # Section 3: Advanced Settings
        advanced_section = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_section)
        advanced_layout.setSpacing(5)
        
        # Add section header
        advanced_header = QtWidgets.QLabel("Advanced Settings")
        advanced_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #58a6ff;")
        advanced_layout.addWidget(advanced_header)
        
        # Create horizontal layouts for paired settings
        freq_layout = QtWidgets.QHBoxLayout()
        freq_layout.setSpacing(10)
        
        # Start Frequency
        start_freq_widget = QtWidgets.QWidget()
        start_freq_layout = QtWidgets.QVBoxLayout(start_freq_widget)
        start_freq_layout.setSpacing(3)
        start_freq_layout.setContentsMargins(0, 0, 0, 0)
        self.start_frequency_entry = QtWidgets.QLineEdit("20")
        self.start_frequency_entry.setStyleSheet("QLineEdit { padding: 5px; border: 1px solid #3d3d3d; border-radius: 3px; }")
        start_freq_layout.addWidget(QtWidgets.QLabel("Start Frequency (Hz):"))
        start_freq_layout.addWidget(self.start_frequency_entry)
        freq_layout.addWidget(start_freq_widget)
        
        # Stop Frequency
        stop_freq_widget = QtWidgets.QWidget()
        stop_freq_layout = QtWidgets.QVBoxLayout(stop_freq_widget)
        stop_freq_layout.setSpacing(3)
        stop_freq_layout.setContentsMargins(0, 0, 0, 0)
        self.stop_frequency_entry = QtWidgets.QLineEdit("20000")
        self.stop_frequency_entry.setStyleSheet("QLineEdit { padding: 5px; border: 1px solid #3d3d3d; border-radius: 3px; }")
        stop_freq_layout.addWidget(QtWidgets.QLabel("Stop Frequency (Hz):"))
        stop_freq_layout.addWidget(self.stop_frequency_entry)
        freq_layout.addWidget(stop_freq_widget)
        
        advanced_layout.addLayout(freq_layout)
        
        # Reference and Steps in another horizontal layout
        ref_steps_layout = QtWidgets.QHBoxLayout()
        ref_steps_layout.setSpacing(10)
        
        # Reference
        ref_widget = QtWidgets.QWidget()
        ref_layout = QtWidgets.QVBoxLayout(ref_widget)
        ref_layout.setSpacing(3)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_entry = QtWidgets.QLineEdit("100")
        self.reference_entry.setStyleSheet("QLineEdit { padding: 5px; border: 1px solid #3d3d3d; border-radius: 3px; }")
        ref_layout.addWidget(QtWidgets.QLabel("Reference in Ohms:"))
        ref_layout.addWidget(self.reference_entry)
        ref_steps_layout.addWidget(ref_widget)
        
        # Steps
        steps_widget = QtWidgets.QWidget()
        steps_layout = QtWidgets.QVBoxLayout(steps_widget)
        steps_layout.setSpacing(3)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        self.step_entry = QtWidgets.QLineEdit("501")
        self.step_entry.setStyleSheet("QLineEdit { padding: 5px; border: 1px solid #3d3d3d; border-radius: 3px; }")
        steps_layout.addWidget(QtWidgets.QLabel("STEPS:"))
        steps_layout.addWidget(self.step_entry)
        ref_steps_layout.addWidget(steps_widget)
        
        advanced_layout.addLayout(ref_steps_layout)
        
        # Section 4: Environmental Readings
        readings_section = QtWidgets.QWidget()
        readings_layout = QtWidgets.QVBoxLayout(readings_section)
        
        # Add section header
        readings_header = QtWidgets.QLabel("Environmental Readings")
        readings_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #58a6ff;")
        readings_layout.addWidget(readings_header)
        
        # Create a grid layout for readings
        readings_grid = QtWidgets.QGridLayout()
        readings_grid.setSpacing(10)
        
        # Style for reading values
        value_style = "font-weight: bold; color: #58a6ff;"
        
        self.temperature_label = QtWidgets.QLabel("Temperature:")
        self.temperature_value = QtWidgets.QLabel("N/A")
        self.temperature_value.setStyleSheet(value_style)
        readings_grid.addWidget(self.temperature_label, 0, 0)
        readings_grid.addWidget(self.temperature_value, 0, 1)
        
        self.humidity_label = QtWidgets.QLabel("Humidity:")
        self.humidity_value = QtWidgets.QLabel("N/A")
        self.humidity_value.setStyleSheet(value_style)
        readings_grid.addWidget(self.humidity_label, 1, 0)
        readings_grid.addWidget(self.humidity_value, 1, 1)
        
        self.pressure_label = QtWidgets.QLabel("Pressure:")
        self.pressure_value = QtWidgets.QLabel("N/A")
        self.pressure_value.setStyleSheet(value_style)
        readings_grid.addWidget(self.pressure_label, 2, 0)
        readings_grid.addWidget(self.pressure_value, 2, 1)
        
        readings_layout.addLayout(readings_grid)
        
        # Add all sections to the main layout
        self.measurement_layout.addWidget(control_section)
        self.measurement_layout.addWidget(config_section)
        self.measurement_layout.addWidget(advanced_section)
        self.measurement_layout.addWidget(readings_section)
        
        # Status section at the bottom
        status_section = QtWidgets.QWidget()
        status_layout = QtWidgets.QVBoxLayout(status_section)
        
        self.measurement_status = QtWidgets.QLabel("Ready for measurement")
        self.measurement_status.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #2ea043;
            padding: 5px;
            border: 1px solid #3d3d3d;
            border-radius: 3px;
            background-color: #2d2d2d;
        """)
        self.measurement_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.measurement_status)
        
        self.measurement_layout.addWidget(status_section)
        
        # Add the group box to the panel
        self.measurement_panel.addWidget(self.measurement_group)
        
        # Add a spacer to push everything to the top
        self.measurement_panel.addStretch()

    def setup_prediction_panel(self):
        # Create a group box for prediction settings
        self.prediction_group = QtWidgets.QGroupBox("Prediction Settings")
        self.prediction_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        self.prediction_layout = QtWidgets.QVBoxLayout(self.prediction_group)
        self.prediction_layout.setSpacing(5)  # Reduce spacing
        self.prediction_layout.setContentsMargins(5, 8, 5, 5)  # Reduce margins
        
        # Section 1: File Selection
        file_section = QtWidgets.QWidget()
        file_layout = QtWidgets.QVBoxLayout(file_section)
        file_layout.setSpacing(5)
        
        # Add section header
        file_header = QtWidgets.QLabel("File Selection")
        file_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #58a6ff;")
        file_layout.addWidget(file_header)
        
        # Load CSV button with improved styling
        self.load_csv_button = QtWidgets.QPushButton("Load CSV File")
        self.load_csv_button.setStyleSheet("""
            QPushButton {
                background-color: #1f6feb;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #388bfd;
            }
        """)
        self.load_csv_button.clicked.connect(self.load_csv_file)
        file_layout.addWidget(self.load_csv_button)
        
        # Section 2: Model Configuration
        model_section = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(model_section)
        model_layout.setSpacing(5)
        
        # Add section header
        model_header = QtWidgets.QLabel("Model Configuration")
        model_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #58a6ff;")
        model_layout.addWidget(model_header)
        
        # Model path with browse button
        path_widget = QtWidgets.QWidget()
        path_layout = QtWidgets.QHBoxLayout(path_widget)
        path_layout.setSpacing(5)
        
        self.model_path_entry = QtWidgets.QLineEdit("")
        self.model_path_entry.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
        """)
        path_layout.addWidget(self.model_path_entry)
        
        self.browse_model_button = QtWidgets.QPushButton("Browse")
        self.browse_model_button.setStyleSheet("""
            QPushButton {
                background-color: #484f58;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                border: none;
            }
            QPushButton:hover {
                background-color: #6e7681;
            }
        """)
        self.browse_model_button.clicked.connect(self.browse_model_placeholder)
        path_layout.addWidget(self.browse_model_button)
        
        model_layout.addWidget(QtWidgets.QLabel("Model Path:"))
        model_layout.addWidget(path_widget)
        
        # Model type selection with improved styling
        self.model_types = ["DNet", "CNet"]
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItems(self.model_types)
        self.model_type_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
        """)
        model_layout.addWidget(QtWidgets.QLabel("Model Type:"))
        model_layout.addWidget(self.model_type_combo)
        
        # Speaker differentiation checkbox with improved styling
        self.speaker_diff_checkbox = QtWidgets.QCheckBox("Enable Speaker Differentiation")
        self.speaker_diff_checkbox.setStyleSheet("""
            QCheckBox {
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
        """)
        self.speaker_diff_checkbox.setChecked(False)
        model_layout.addWidget(self.speaker_diff_checkbox)
        
        # Add sections to the prediction layout
        self.prediction_layout.addWidget(file_section)
        self.prediction_layout.addWidget(model_section)
        
        # Add the prediction group to the panel
        self.prediction_panel.addWidget(self.prediction_group)
        
        # Create a group box for prediction results
        self.results_group = QtWidgets.QGroupBox("Prediction Results")
        self.results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        self.results_layout = QtWidgets.QVBoxLayout(self.results_group)
        self.results_layout.setSpacing(5)  # Reduce spacing
        self.results_layout.setContentsMargins(5, 8, 5, 5)  # Reduce margins
        
        # Results section
        results_section = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_section)
        results_layout.setSpacing(10)
        
        # Prediction result with improved styling
        self.prediction_result_label = QtWidgets.QLabel("No prediction yet")
        self.prediction_result_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
            padding: 5px;
            background-color: #2d2d2d;
            border-radius: 5px;
        """)
        self.prediction_result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.prediction_result_label)
        
        # Confidence label with improved styling
        self.confidence_label = QtWidgets.QLabel("Confidence: N/A")
        self.confidence_label.setStyleSheet("""
            font-size: 16px;
            color: #8b949e;
            padding: 5px;
        """)
        self.confidence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.confidence_label)
        
        # CSV file path with improved styling
        path_label = QtWidgets.QLabel("CSV File:")
        path_label.setStyleSheet("color: #8b949e;")
        results_layout.addWidget(path_label)
        
        self.csv_path_label = QtWidgets.QLabel("No CSV file loaded")
        self.csv_path_label.setStyleSheet("""
            color: #8b949e;
            padding: 5px;
            background-color: #2d2d2d;
            border-radius: 3px;
        """)
        self.csv_path_label.setWordWrap(True)
        self.csv_path_label.setMaximumWidth(350)
        self.csv_path_label.setMinimumHeight(40)
        results_layout.addWidget(self.csv_path_label)
        
        # Predict button with improved styling
        self.predict_button = QtWidgets.QPushButton("Predict Length")
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #1f6feb;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #388bfd;
            }
        """)
        self.predict_button.clicked.connect(self.predict_length_placeholder)
        results_layout.addWidget(self.predict_button)
        
        # Prediction status with improved styling
        self.prediction_status = QtWidgets.QLabel("Ready for prediction")
        self.prediction_status.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #58a6ff;
            padding: 5px;
            border: 1px solid #3d3d3d;
            border-radius: 3px;
            background-color: #2d2d2d;
        """)
        self.prediction_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.prediction_status)
        
        # Add the results section to the results group
        self.results_layout.addWidget(results_section)
        
        # Add the results group to the panel
        self.prediction_panel.addWidget(self.results_group)
        
        # Add a spacer to push everything to the top
        self.prediction_panel.addStretch()

    def load_dwf(self):
        if sys.platform.startswith("win"):
            self.dwf = cdll.LoadLibrary("dwf.dll")
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")

    def start_measurement(self):
        selected_type = self.type_combo.currentText()
        selected_length = self.length_combo.currentText()
        start_freq = self.start_frequency_entry.text()
        stop_freq = self.stop_frequency_entry.text()
        reference = self.reference_entry.text()
        steps = self.step_entry.text()
        
        # Validate inputs
        if not selected_type or not selected_length:
            QtWidgets.QMessageBox.critical(self, "Error", "Please select type and length.")
            return
        
        # Update measurement status
        self.measurement_status.setText("Measurement in progress...")
        self.measurement_status.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        
        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        
        # Start data collection in a new thread
        thread = threading.Thread(target=self.collect_data, args=(folder_name, start_freq, stop_freq, reference, steps))
        thread.start()
        
        self.start_button.setStyleSheet("background-color: red; color: black; font-size: 18px;")
        self.start_button.setText("Running")
        
        # Start the timer to read Arduino data every second
        self.timer.start(1000)

    def read_arduino_data(self):
        if self.serial_port and self.serial_port.in_waiting > 0:
            line = self.serial_port.readline().decode('utf-8').rstrip()
            try:
                # Expecting "temp,humidity,pressure" format
                temperature, humidity, pressure = map(float, line.split(','))
                self.temperature_value.setText(f"{temperature:.2f}")
                self.humidity_value.setText(f"{humidity:.2f}")
                self.pressure_value.setText(f"{pressure:.2f}")
                return temperature, humidity, pressure
            except ValueError:
                print("Invalid data from Arduino: ", line)  # Log the invalid data for debugging

    def collect_data(self, folder_name, start_freq, stop_freq, reference, steps):
        # Opens the device, Analog Discovery 2 through the serial port.
        hdwf = c_int()
        szerr = create_string_buffer(512)
        self.dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
        
        if hdwf.value == hdwfNone.value:
            self.dwf.FDwfGetLastErrorMsg(szerr)
            self.update_progress(f"Failed to open device: {str(szerr.value)}")
            return
        
        # Impedance Measurement settings
        self.dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3))
        sts = c_byte()
        steps = int(steps)
        start = int(start_freq)
        stop = int(stop_freq)
        reference = int(reference)
        self.update_progress("Status: Running measurement")
        
        self.dwf.FDwfAnalogImpedanceReset(hdwf)
        self.dwf.FDwfAnalogImpedanceModeSet(hdwf, c_int(0))
        self.dwf.FDwfAnalogImpedanceReferenceSet(hdwf, c_double(reference))
        self.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(start))
        self.dwf.FDwfAnalogImpedanceAmplitudeSet(hdwf, c_double(0.5))
        self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(1))
        time.sleep(0.5)
        
        rgHz = [0.0] * steps
        rgZ = [0.0] * steps
        rgRs = [0.0] * steps
        rgXs = [0.0] * steps
        
        file_path = os.path.join(self.base_folder, f"{folder_name}_Run1.csv")
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Frequency (Hz)", "Trace Rs (Ohm)", "Trace Xs (Ohm)", "Temperature (°C)", "Humidity (%)", "Pressure (hPa)"])
            
            for i in range(steps):
                hz = start + i * (stop - start) / (steps - 1)
                rgHz[i] = hz
                self.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(hz))
                
                while True:
                    if self.dwf.FDwfAnalogImpedanceStatus(hdwf, byref(sts)) == 0:
                        self.dwf.FDwfGetLastErrorMsg(szerr)
                        print(str(szerr.value))
                        return
                    if sts.value == 2:
                        break
                
                resistance = c_double()
                reactance = c_double()
                self.dwf.FDwfAnalogImpedanceStatusMeasure(hdwf, DwfAnalogImpedanceResistance, byref(resistance))
                self.dwf.FDwfAnalogImpedanceStatusMeasure(hdwf, DwfAnalogImpedanceReactance, byref(reactance))
                rgRs[i] = resistance.value
                rgXs[i] = reactance.value
                rgZ[i] = (resistance.value**2 + reactance.value**2)**0.5
                
                writer.writerow([hz, rgRs[i], rgXs[i], self.temperature_value.text(), self.humidity_value.text(), self.pressure_value.text()])
                self.update_progress(f"Step {i + 1}/{steps}: Frequency {hz:.2f} Hz, Impedance {rgZ[i]:.2f} Ohms")
        
        self.frequencies = rgHz
        self.impedance = rgZ
        self.resistance = rgRs
        self.reactance = rgXs
        self.csv_path = file_path
        
        # Update the CSV path label
        self.csv_path_label.setText(file_path)
        
        # Update the last measurement path
        self.last_measurement_path.setText(file_path)
        
        self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.dwf.FDwfDeviceClose(hdwf)
        
        # Stop the timer after data collection is complete
        self.timer.stop()
        self.update_progress("Measurement completed.")
        self.start_button.setText("Start Measurement")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 18px;")
        
        # Update measurement status
        self.measurement_status.setText("Measurement completed")
        self.measurement_status.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")

    def update_progress(self, message):
        self.progress_text.append(message)
        self.progress_text.verticalScrollBar().setValue(self.progress_text.verticalScrollBar().maximum())

    def load_csv_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.csv_path = file_path
            self.csv_path_label.setText(file_path)
            self.update_progress(f"CSV file loaded: {file_path}")
            
            # Update prediction status
            self.prediction_status.setText("CSV loaded - Ready for prediction")
            self.prediction_status.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")

    # Placeholder methods for prediction functionality (to be implemented later)
    def browse_model_placeholder(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Model File", "", "Keras Models (*.keras *.h5)")
        if file_path:
            self.model_path_entry.setText(file_path)
            self.update_progress(f"Model path set to: {file_path} (functionality not implemented)")

    def predict_length_placeholder(self):
        self.update_progress("Prediction functionality has been removed and will be implemented later.")
        QtWidgets.QMessageBox.information(self, "Not Implemented", "The prediction functionality has been removed and will be implemented later.")
        
        # Update prediction status
        self.prediction_status.setText("Prediction not implemented")
        self.prediction_status.setStyleSheet("font-size: 14px; font-weight: bold; color: orange;")

    def closeEvent(self, event):
        if self.serial_port:
            self.serial_port.close()  # Close the serial port when the application is closed
        event.accept()

    def create_speaker_icon(self, icon_path):
        """Create a simple speaker icon and save it as PNG."""
        # Create a 64x64 image with transparent background
        icon_size = 64
        image = QtGui.QImage(icon_size, icon_size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        
        # Create painter
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Set up the pen and brush
        painter.setPen(QtGui.QPen(QtGui.QColor("#58a6ff"), 2))  # Use the theme's blue color
        
        # Calculate center and dimensions
        center_x = icon_size // 2
        center_y = icon_size // 2
        
        # Draw outer circle (speaker frame)
        frame_radius = 24
        painter.setBrush(QtGui.QBrush())  # No fill for frame
        painter.drawEllipse(center_x - frame_radius, center_y - frame_radius, 
                          frame_radius * 2, frame_radius * 2)
        
        # Draw middle circle (suspension)
        suspension_radius = 18
        painter.drawEllipse(center_x - suspension_radius, center_y - suspension_radius, 
                          suspension_radius * 2, suspension_radius * 2)
        
        # Draw inner circle (dome/cone)
        dome_radius = 12
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#58a6ff")))  # Fill for dome
        painter.drawEllipse(center_x - dome_radius, center_y - dome_radius, 
                          dome_radius * 2, dome_radius * 2)
        
        # Draw sound waves
        painter.setBrush(QtGui.QBrush())  # No fill for waves
        wave_pen = QtGui.QPen(QtGui.QColor("#58a6ff"))
        
        # Draw three waves with decreasing width
        for i, offset in enumerate([32, 38, 44]):
            wave_pen.setWidth(3 - i)  # Decreasing width for outer waves
            painter.setPen(wave_pen)
            
            # Draw left wave
            start_angle = 150 * 16  # 150 degrees * 16 (Qt angle units)
            span_angle = 60 * 16    # 60 degrees span
            painter.drawArc(center_x - offset, center_y - offset, 
                          offset * 2, offset * 2, 
                          start_angle, span_angle)
            
            # Draw right wave
            start_angle = -30 * 16  # -30 degrees
            painter.drawArc(center_x - offset, center_y - offset, 
                          offset * 2, offset * 2, 
                          start_angle, span_angle)
        
        painter.end()
        
        # Save the image
        image.save(icon_path)

# Main function that starts the program
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImpedancePredictionApp()
    window.show()
    sys.exit(app.exec())
