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
from PyQt6 import QtWidgets, QtCore
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
        self.setGeometry(100, 100, 800, 600)  # Reduced size since we're removing the plot

        # Main layout (vertical)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        
        # Create a horizontal layout for the main content
        self.content_layout = QtWidgets.QHBoxLayout()
        
        # Progress text box at the bottom - CREATE THIS FIRST so update_progress works
        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
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
        self.measurement_layout = QtWidgets.QVBoxLayout(self.measurement_group)
        
        # Start Button
        self.start_button = QtWidgets.QPushButton("Start Measurement")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 18px;")
        self.start_button.clicked.connect(self.start_measurement)
        self.measurement_layout.addWidget(self.start_button)
        
        # Add some spacing after the start button
        self.measurement_layout.addSpacing(5)
        
        # Dropdown for earphone selection
        self.types = ["A", "B", "C", "D"]
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(self.types)
        self.measurement_layout.addWidget(QtWidgets.QLabel("Earphone Type:"))
        self.measurement_layout.addWidget(self.type_combo)
        
        # Dropdown for length selection
        self.lengths = [str(i) for i in range(5, 31, 3)] + [str(i) for i in [9, 24, 39]] + ["Open", "Blocked"]
        self.length_combo = QtWidgets.QComboBox()
        self.length_combo.addItems(self.lengths)
        self.measurement_layout.addWidget(QtWidgets.QLabel("Length (mm):"))
        self.measurement_layout.addWidget(self.length_combo)
        
        # Frequency entries
        self.start_frequency_entry = QtWidgets.QLineEdit("20")
        self.stop_frequency_entry = QtWidgets.QLineEdit("20000")
        self.measurement_layout.addWidget(QtWidgets.QLabel("Start Frequency (Hz):"))
        self.measurement_layout.addWidget(self.start_frequency_entry)
        self.measurement_layout.addWidget(QtWidgets.QLabel("Stop Frequency (Hz):"))
        self.measurement_layout.addWidget(self.stop_frequency_entry)
        
        # Reference resistor value
        self.reference_entry = QtWidgets.QLineEdit("100")
        self.measurement_layout.addWidget(QtWidgets.QLabel("Reference in Ohms:"))
        self.measurement_layout.addWidget(self.reference_entry)
        
        # Steps entry
        self.step_entry = QtWidgets.QLineEdit("501")
        self.measurement_layout.addWidget(QtWidgets.QLabel("STEPS:"))
        self.measurement_layout.addWidget(self.step_entry)
        
        # Create a horizontal layout for temperature, humidity, and pressure
        self.readings_layout = QtWidgets.QHBoxLayout()
        
        # Add fields for temperature, humidity, and pressure
        self.humidity_label = QtWidgets.QLabel("Humidity (%):")
        self.humidity_value = QtWidgets.QLabel("N/A")
        self.readings_layout.addWidget(self.humidity_label)
        self.readings_layout.addWidget(self.humidity_value)
        
        self.temperature_label = QtWidgets.QLabel("Temperature (°C):")
        self.temperature_value = QtWidgets.QLabel("N/A")
        self.readings_layout.addWidget(self.temperature_label)
        self.readings_layout.addWidget(self.temperature_value)
        
        self.pressure_label = QtWidgets.QLabel("Pressure (hPa):")
        self.pressure_value = QtWidgets.QLabel("N/A")
        self.readings_layout.addWidget(self.pressure_label)
        self.readings_layout.addWidget(self.pressure_value)
        
        # Add the readings layout to the measurement layout
        self.measurement_layout.addLayout(self.readings_layout)
        
        # Add more spacing to match the height of the prediction panels
        self.measurement_layout.addSpacing(30)
        
        # Add a "Last Measurement" section
        self.last_measurement_label = QtWidgets.QLabel("Last Measurement:")
        self.last_measurement_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.measurement_layout.addWidget(self.last_measurement_label)
        
        # Add a label to display the last measurement file path
        self.last_measurement_path = QtWidgets.QLabel("No measurement taken yet")
        self.last_measurement_path.setWordWrap(True)
        self.last_measurement_path.setMinimumHeight(40)
        self.measurement_layout.addWidget(self.last_measurement_path)
        
        # Add some spacing
        self.measurement_layout.addSpacing(10)
        
        # Add a "Ready for measurement" status label at the bottom
        self.measurement_status = QtWidgets.QLabel("Ready for measurement")
        self.measurement_status.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        self.measurement_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.measurement_layout.addWidget(self.measurement_status)
        
        # Add the group box to the panel
        self.measurement_panel.addWidget(self.measurement_group)
        
        # Add a spacer to push everything to the top
        self.measurement_panel.addStretch()

    def setup_prediction_panel(self):
        # Create a group box for prediction settings (UI only)
        self.prediction_group = QtWidgets.QGroupBox("Prediction Settings (UI Only)")
        self.prediction_layout = QtWidgets.QVBoxLayout(self.prediction_group)
        
        # Load CSV button
        self.load_csv_button = QtWidgets.QPushButton("Load CSV File")
        self.load_csv_button.setStyleSheet("font-size: 14px;")
        self.load_csv_button.clicked.connect(self.load_csv_file)
        self.prediction_layout.addWidget(self.load_csv_button)
        
        # Add some spacing
        self.prediction_layout.addSpacing(5)
        
        # Model selection
        self.prediction_layout.addWidget(QtWidgets.QLabel("Model Path:"))
        self.model_path_entry = QtWidgets.QLineEdit("")
        self.prediction_layout.addWidget(self.model_path_entry)
        
        # Browse button for model
        self.browse_model_button = QtWidgets.QPushButton("Browse")
        self.browse_model_button.clicked.connect(self.browse_model_placeholder)
        self.prediction_layout.addWidget(self.browse_model_button)
        
        # Add some spacing
        self.prediction_layout.addSpacing(5)
        
        # Model type selection
        self.model_types = ["DNet", "CNet"]
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItems(self.model_types)
        self.prediction_layout.addWidget(QtWidgets.QLabel("Model Type:"))
        self.prediction_layout.addWidget(self.model_type_combo)
        
        # Speaker differentiation checkbox
        self.speaker_diff_checkbox = QtWidgets.QCheckBox("Enable Speaker Differentiation")
        self.speaker_diff_checkbox.setChecked(False)
        self.prediction_layout.addWidget(self.speaker_diff_checkbox)
        
        # Add the group box to the panel
        self.prediction_panel.addWidget(self.prediction_group)
        
        # Create a group box for prediction results
        self.results_group = QtWidgets.QGroupBox("Prediction Results (UI Only)")
        self.results_layout = QtWidgets.QVBoxLayout(self.results_group)
        
        # Prediction results
        self.prediction_result_label = QtWidgets.QLabel("No prediction yet")
        self.prediction_result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.prediction_result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.prediction_result_label)
        
        # Confidence label
        self.confidence_label = QtWidgets.QLabel("Confidence: N/A")
        self.confidence_label.setStyleSheet("font-size: 16px;")
        self.confidence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.confidence_label)
        
        # Add some spacing
        self.results_layout.addSpacing(10)
        
        # CSV file path label with word wrap enabled
        self.results_layout.addWidget(QtWidgets.QLabel("CSV File:"))
        self.csv_path_label = QtWidgets.QLabel("No CSV file loaded")
        self.csv_path_label.setWordWrap(True)  # Enable word wrap
        self.csv_path_label.setMaximumWidth(350)  # Set maximum width to force wrapping
        self.csv_path_label.setMinimumHeight(40)  # Set minimum height to ensure space for wrapped text
        self.results_layout.addWidget(self.csv_path_label)
        
        # Add some spacing
        self.results_layout.addSpacing(10)
        
        # Manual predict button
        self.predict_button = QtWidgets.QPushButton("Predict Length")
        self.predict_button.setStyleSheet("background-color: blue; color: white; font-size: 16px;")
        self.predict_button.clicked.connect(self.predict_length_placeholder)
        self.results_layout.addWidget(self.predict_button)
        
        # Add a prediction status label
        self.prediction_status = QtWidgets.QLabel("Ready for prediction")
        self.prediction_status.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        self.prediction_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.prediction_status)
        
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

# Main function that starts the program
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImpedancePredictionApp()
    window.show()
    sys.exit(app.exec())
