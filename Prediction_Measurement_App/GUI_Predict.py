"""
Name: Impedance Measurement and Length Prediction GUI
Author: Max Chen
Date: March 1, 2024
Description: 
This is a GUI application that combines impedance measurement using the Analog Discovery 2
with length prediction using a trained neural network model. The application allows users
to perform impedance measurements and then predict the length of the earphone tube.
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import serial
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path to import dwfconstants
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "Analog Discovery"))
from dwfconstants import *

class ImpedancePredictionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earphones Impedance Measurement and Length Prediction")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize serial communication for Arduino (temperature, humidity, pressure)
        self.serial_port = None
        try:
            self.serial_port = serial.Serial('COM8', 115200, timeout=1)  # Adjust COM port as necessary
        except serial.SerialException:
            pass  # We'll handle it later

        # Main layout (horizontal)
        self.main_layout = QtWidgets.QHBoxLayout(self)

        # Create left panel for controls
        self.left_panel = QtWidgets.QVBoxLayout()
        
        # Create tabs for measurement and prediction
        self.tabs = QtWidgets.QTabWidget()
        self.measurement_tab = QtWidgets.QWidget()
        self.prediction_tab = QtWidgets.QWidget()
        
        self.tabs.addTab(self.measurement_tab, "Measurement")
        self.tabs.addTab(self.prediction_tab, "Prediction")
        
        # Setup measurement tab
        self.setup_measurement_tab()
        
        # Setup prediction tab
        self.setup_prediction_tab()
        
        # Add tabs to left panel
        self.left_panel.addWidget(self.tabs)
        
        # Add left panel to main layout
        self.main_layout.addLayout(self.left_panel, stretch=1)
        
        # Create right panel for plots
        self.right_panel = QtWidgets.QVBoxLayout()
        
        # Create initial plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.right_panel.addWidget(self.canvas)
        
        # Add right panel to main layout
        self.main_layout.addLayout(self.right_panel, stretch=2)
        
        # Load DWF library
        self.load_dwf()
        
        # Initialize data storage
        self.measurement_data = None
        self.frequencies = None
        self.impedance = None
        self.resistance = None
        self.reactance = None
        
        # Update progress based on Arduino connection status
        if self.serial_port:
            self.update_progress("Arduino connected.")
        else:
            self.update_progress("Warning: Arduino not connected. Data collection will not include temperature, humidity, and pressure.")
        
        # Timer for reading Arduino data
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.read_arduino_data)
        
        # Load model
        self.model = None
        
        # Look for model in the local Model directory first
        model_dir = os.path.join(current_dir, "Model")
        if os.path.exists(model_dir):
            # Find the first .keras or .h5 file in the Model directory
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))]
            if model_files:
                self.model_path = os.path.join(model_dir, model_files[0])
                self.update_progress(f"Found model in local Model directory: {model_files[0]}")
            else:
                # Fallback to the model in the parent directory
                self.model_path = os.path.join(parent_dir, "Impedance-main", "best_model.keras")
                self.update_progress("No model found in local Model directory. Using default path.")
        else:
            # Fallback to the model in the parent directory
            self.model_path = os.path.join(parent_dir, "Impedance-main", "best_model.keras")
            self.update_progress("Model directory not found. Using default path.")
        
        # Try to load the model
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                self.update_progress(f"Model loaded from {self.model_path}")
            except Exception as e:
                self.update_progress(f"Error loading model: {str(e)}")
        else:
            self.update_progress(f"Model not found at {self.model_path}")

    def setup_measurement_tab(self):
        # Create layout for measurement tab
        self.measurement_layout = QtWidgets.QVBoxLayout(self.measurement_tab)
        
        # Start Button
        self.start_button = QtWidgets.QPushButton("Start Measurement")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 24px;")
        self.start_button.clicked.connect(self.start_measurement)
        self.measurement_layout.addWidget(self.start_button)
        
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
        
        # Progress text box
        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.measurement_layout.addWidget(QtWidgets.QLabel("Status:"))
        self.measurement_layout.addWidget(self.progress_text)

    def setup_prediction_tab(self):
        # Create layout for prediction tab
        self.prediction_layout = QtWidgets.QVBoxLayout(self.prediction_tab)
        
        # Predict button
        self.predict_button = QtWidgets.QPushButton("Predict Length")
        self.predict_button.setStyleSheet("background-color: blue; color: white; font-size: 24px;")
        self.predict_button.clicked.connect(self.predict_length)
        self.prediction_layout.addWidget(self.predict_button)
        
        # Load CSV button
        self.load_csv_button = QtWidgets.QPushButton("Load CSV File")
        self.load_csv_button.clicked.connect(self.load_csv_file)
        self.prediction_layout.addWidget(self.load_csv_button)
        
        # Model selection
        self.model_path_entry = QtWidgets.QLineEdit(self.model_path if hasattr(self, 'model_path') else "")
        self.prediction_layout.addWidget(QtWidgets.QLabel("Model Path:"))
        self.prediction_layout.addWidget(self.model_path_entry)
        
        # Browse button for model
        self.browse_model_button = QtWidgets.QPushButton("Browse")
        self.browse_model_button.clicked.connect(self.browse_model)
        self.prediction_layout.addWidget(self.browse_model_button)
        
        # Model type selection
        self.model_types = ["DNet", "CNet"]
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItems(self.model_types)
        self.prediction_layout.addWidget(QtWidgets.QLabel("Model Type:"))
        self.prediction_layout.addWidget(self.model_type_combo)
        
        # Speaker differentiation checkbox
        self.speaker_diff_checkbox = QtWidgets.QCheckBox("Enable Speaker Differentiation")
        self.prediction_layout.addWidget(self.speaker_diff_checkbox)
        
        # Prediction results
        self.prediction_layout.addWidget(QtWidgets.QLabel("Prediction Results:"))
        self.prediction_result_label = QtWidgets.QLabel("No prediction yet")
        self.prediction_result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.prediction_layout.addWidget(self.prediction_result_label)
        
        # Confidence label
        self.confidence_label = QtWidgets.QLabel("Confidence: N/A")
        self.prediction_layout.addWidget(self.confidence_label)
        
        # CSV file path label
        self.csv_path_label = QtWidgets.QLabel("No CSV file loaded")
        self.prediction_layout.addWidget(QtWidgets.QLabel("CSV File:"))
        self.prediction_layout.addWidget(self.csv_path_label)
        
        # Add spacer to push everything to the top
        self.prediction_layout.addStretch()

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
        
        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        
        # Start data collection in a new thread
        thread = threading.Thread(target=self.collect_data, args=(folder_name, start_freq, stop_freq, reference, steps))
        thread.start()
        
        self.start_button.setStyleSheet("background-color: red; color: black; font-size: 24px;")
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
        
        # Update the plot
        self.update_plot(rgHz, rgZ)
        
        # Update the CSV path label
        self.csv_path_label.setText(file_path)
        
        self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.dwf.FDwfDeviceClose(hdwf)
        
        # Stop the timer after data collection is complete
        self.timer.stop()
        self.update_progress("Measurement completed.")
        self.start_button.setText("Start Measurement")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 24px;")
        
        # Switch to prediction tab
        self.tabs.setCurrentIndex(1)

    def update_progress(self, message):
        self.progress_text.append(message)
        self.progress_text.verticalScrollBar().setValue(self.progress_text.verticalScrollBar().maximum())

    def update_plot(self, frequencies, impedance):
        self.ax.clear()
        self.ax.plot(frequencies, impedance, label='|Z| (Ohms)')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Impedance (Ohms)')
        self.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.ax.legend()
        self.canvas.draw()

    def load_csv_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.csv_path = file_path
            self.csv_path_label.setText(file_path)
            self.update_progress(f"CSV file loaded: {file_path}")

    def browse_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Model File", "", "Keras Models (*.keras *.h5)")
        if file_path:
            self.model_path_entry.setText(file_path)
            try:
                self.model = load_model(file_path)
                self.update_progress(f"Model loaded: {file_path}")
            except Exception as e:
                self.update_progress(f"Error loading model: {str(e)}")

    def min_max(self, col):
        """Normalize column using min-max scaling"""
        return (col - min(col)) / (max(col) - min(col))

    def preprocess_csv(self, file_path, columns=None, start=0, end=500):
        """Preprocess a CSV file for prediction"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Default columns if not specified
        if columns is None:
            columns = ["PH", "MAG", "RS", "XS", "REC"]
        
        # Read CSV file
        import pandas as pd
        data = pd.read_csv(file_path, header=0)
        
        # Extract and preprocess columns
        processed_cols = []
        for col in columns:
            col = col.upper()
            if col == "FREQ":
                freq = self.min_max(data.iloc[:, 0][start:end+1])
                processed_cols.append(freq)
            elif col == "PH":
                # Calculate phase from Rs and Xs
                rs = data.iloc[:, 1][start:end+1]
                xs = data.iloc[:, 2][start:end+1]
                phase = np.arctan2(xs, rs)
                phase = self.min_max(phase)
                processed_cols.append(phase)
            elif col == "MAG":
                # Calculate magnitude from Rs and Xs
                rs = data.iloc[:, 1][start:end+1]
                xs = data.iloc[:, 2][start:end+1]
                mag = np.sqrt(rs**2 + xs**2)
                mag = self.min_max(mag)
                processed_cols.append(mag)
            elif col == "RS":
                rs = self.min_max(data.iloc[:, 1][start:end+1])
                processed_cols.append(rs)
            elif col == "XS":
                xs = self.min_max(data.iloc[:, 2][start:end+1])
                processed_cols.append(xs)
            elif col == "REC":
                rs = self.min_max(data.iloc[:, 1][start:end+1])
                xs = self.min_max(data.iloc[:, 2][start:end+1])
                rec = rs + 1j * xs
                processed_cols.append(rec)
        
        # Reshape for model input
        input_data = np.array([processed_cols])
        
        return input_data

    def create_label_encoder(self, speakers=None, lengths=None, speaker_differentiation=False):
        """Create and fit a label encoder for the model output"""
        # Default values if not specified
        if speakers is None:
            speakers = ["A", "B", "C", "D"]
        if lengths is None:
            lengths = ["5", "8", "9", "11", "14", "17", "20", "23", "24", "26", "29", "39", "Blocked", "Open"]
        
        # Create labels
        labels = []
        for speaker in speakers:
            for length in lengths:
                # Format the length to match the model's output format
                if length in ["5", "8", "9"]:
                    length = "0" + length
                if length == "Blocked":
                    length = "00" + length
                    
                # Create the label
                if speaker_differentiation:
                    label = f"{speaker}_{length}"
                else:
                    label = length
                    
                labels.append(label)
        
        # Create and fit the label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        
        return label_encoder

    def predict_length(self):
        if not hasattr(self, 'csv_path') or not self.csv_path:
            QtWidgets.QMessageBox.critical(self, "Error", "No CSV file loaded. Please load a CSV file or perform a measurement first.")
            return
        
        # Get model path from entry
        model_path = self.model_path_entry.text()
        
        # Check if model exists
        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(self, "Error", f"Model file not found: {model_path}")
            return
        
        # Load model if not already loaded
        if self.model is None:
            try:
                self.model = load_model(model_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
                return
        
        # Get model type
        model_type = self.model_type_combo.currentText()
        
        # Get speaker differentiation setting
        speaker_diff = self.speaker_diff_checkbox.isChecked()
        
        try:
            # Preprocess the CSV file
            columns = ["PH", "MAG", "RS", "XS", "REC"]
            input_data = self.preprocess_csv(self.csv_path, columns)
            
            # Reshape for CNet if needed
            if model_type == "CNet":
                input_data = input_data[:, np.newaxis, :]
            
            # Get the label encoder
            label_encoder = self.create_label_encoder(speaker_differentiation=speaker_diff)
            
            # Make prediction
            prediction = self.model.predict(input_data)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            
            # Get prediction probability
            prediction_probability = np.max(prediction, axis=1)[0]
            
            # Update the prediction result label
            self.prediction_result_label.setText(f"Predicted Length: {predicted_class}")
            self.confidence_label.setText(f"Confidence: {prediction_probability:.4f}")
            
            self.update_progress(f"Prediction: {predicted_class} with confidence {prediction_probability:.4f}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")
            self.update_progress(f"Error during prediction: {str(e)}")

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
