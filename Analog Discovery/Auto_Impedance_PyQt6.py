"""
Name: Auto Impedance V1.1 with PyQt6
Author: Max Chen
Date: February 26, 2025
Description: 
This is a GUI script written with PyQt6 to automate and speed up earphone impedance data collection with the Analog Discovery 2 from Digilent.
Using the API from the DIGILENT WaveForms software, this is a custom script to directly use the existing impedance measuring function in the software.
"""

# Import necessary libraries.
import os
import sys
import time
import csv
import threading
from ctypes import *
from dwfconstants import *
from PyQt6 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import serial

class DataCollectionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earphones Impedance Data Collector")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize serial communication
        self.serial_port = None  # Initialize to None first
        try:
            self.serial_port = serial.Serial('COM3', 9600, timeout=1)  # Adjust COM port as necessary
        except serial.SerialException:
            pass  # Do nothing, we'll handle it later

        # Layouts
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.button_layout = QtWidgets.QVBoxLayout()
        self.input_layout = QtWidgets.QVBoxLayout()

        # Dropdown for earphone selection
        self.types = ["A", "B", "C", "D"]
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(self.types)
        self.input_layout.addWidget(QtWidgets.QLabel("Earphone Type:"))
        self.input_layout.addWidget(self.type_combo)

        # Dropdown for length selection
        self.lengths = [str(i) for i in range(5, 31, 3)] + [str(i) for i in [9, 24, 39]] + ["Open", "Blocked"]
        self.length_combo = QtWidgets.QComboBox()
        self.length_combo.addItems(self.lengths)
        self.input_layout.addWidget(QtWidgets.QLabel("Length (mm):"))
        self.input_layout.addWidget(self.length_combo)

        # Entry for number of repetitions
        self.repetitions_entry = QtWidgets.QLineEdit("100")
        self.input_layout.addWidget(QtWidgets.QLabel("Number of Repetitions:"))
        self.input_layout.addWidget(self.repetitions_entry)

        # Frequency entries
        self.start_frequency_entry = QtWidgets.QLineEdit("20")
        self.stop_frequency_entry = QtWidgets.QLineEdit("20000")
        self.input_layout.addWidget(QtWidgets.QLabel("Start Frequency (Hz):"))
        self.input_layout.addWidget(self.start_frequency_entry)
        self.input_layout.addWidget(QtWidgets.QLabel("Stop Frequency (Hz):"))
        self.input_layout.addWidget(self.stop_frequency_entry)

        # Reference resistor value
        self.reference_entry = QtWidgets.QLineEdit("100")
        self.input_layout.addWidget(QtWidgets.QLabel("Reference in Ohms:"))
        self.input_layout.addWidget(self.reference_entry)

        # Steps entry
        self.step_entry = QtWidgets.QLineEdit("501")
        self.input_layout.addWidget(QtWidgets.QLabel("STEPS:"))
        self.input_layout.addWidget(self.step_entry)

        # Progress text box
        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.input_layout.addWidget(QtWidgets.QLabel("Status:"))
        self.input_layout.addWidget(self.progress_text)

        # Add fields for temperature and humidity
        self.humidity_label = QtWidgets.QLabel("Humidity (%):")
        self.humidity_value = QtWidgets.QLabel("N/A")  # Placeholder for humidity value
        self.input_layout.addWidget(self.humidity_label)
        self.input_layout.addWidget(self.humidity_value)

        self.temperature_label = QtWidgets.QLabel("Temperature (°C):")
        self.temperature_value = QtWidgets.QLabel("N/A")  # Placeholder for temperature value
        self.input_layout.addWidget(self.temperature_label)
        self.input_layout.addWidget(self.temperature_value)

        # Start Button
        self.start_button = QtWidgets.QPushButton("Start Data Collection")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 24px;")
        self.start_button.clicked.connect(self.start_data_collection)
        self.button_layout.addWidget(self.start_button)

        # Add input layout to the main layout
        self.main_layout.addLayout(self.input_layout)

        # Add the button layout to the input layout
        self.input_layout.addLayout(self.button_layout)

        # Create initial plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)

        # Add the plot to the right side of the main layout
        self.main_layout.addWidget(self.canvas)

        # Set the title of the plot
        self.ax.set_title('Impedance Magnitude |Z| vs Frequency')

        # Load DWF library
        self.load_dwf()

        # Update progress based on Arduino connection status
        if self.serial_port:
            self.update_progress("Arduino connected.")
        else:
            self.update_progress("Warning: Arduino not connected. Data collection will not include temperature and humidity.")

    def load_dwf(self):
        if sys.platform.startswith("win"):
            self.dwf = cdll.LoadLibrary("dwf.dll")
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")

    def start_data_collection(self):
        selected_type = self.type_combo.currentText()
        selected_length = self.length_combo.currentText()
        repetitions = self.repetitions_entry.text()
        start_freq = self.start_frequency_entry.text()
        stop_freq = self.stop_frequency_entry.text()
        reference = self.reference_entry.text()
        steps = self.step_entry.text()

        # Validate inputs
        if not selected_type or not selected_length or not repetitions:
            QtWidgets.QMessageBox.critical(self, "Error", "Please select type, length, and number of repetitions.")
            return

        try:
            repetitions = int(repetitions)
            if repetitions <= 0:
                raise ValueError("Number of repetitions must be a positive integer.")
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

        # Start data collection in a new thread
        thread = threading.Thread(target=self.collect_data, args=(folder_name, repetitions, start_freq, stop_freq, reference, steps))
        thread.start()

        self.start_button.setStyleSheet("background-color: red; color: black; font-size: 24px;")

    def read_arduino_data(self):
        if self.serial_port and self.serial_port.in_waiting > 0:
            line = self.serial_port.readline().decode('utf-8').rstrip()
            try:
                temperature, humidity = map(float, line.split(','))  # Expecting "temp,humidity" format
                self.temperature_value.setText(f"{temperature:.2f}")
                self.humidity_value.setText(f"{humidity:.2f}")
                return temperature, humidity
            except ValueError:
                print("Invalid data from Arduino")

    def collect_data(self, folder_name, repetitions, start_freq, stop_freq, reference, steps):
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
        self.update_progress("Status: Running")
        self.start_button.setText("Running")

        for run in range(repetitions):
            # Read temperature and humidity from Arduino
            temperature, humidity = (None, None)
            if self.serial_port:
                temperature, humidity = self.read_arduino_data()

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

            file_path = os.path.join(self.base_folder, f"{folder_name}_Run{run + 1}.csv")
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Frequency (Hz)", "Trace Rs (Ohm)", "Trace Xs (Ohm)", "Temperature (°C)", "Humidity (%)"])

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

                    writer.writerow([hz, rgRs[i], rgXs[i], temperature, humidity])
                    self.update_progress(f"Run {run + 1}, Step {i + 1}/{steps}: Frequency {hz:.2f} Hz, Impedance {rgZ[i]:.2f} Ohms")

            self.update_plot(rgHz, rgZ)
            time.sleep(0.5)

        self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.dwf.FDwfDeviceClose(hdwf)
        self.update_progress("Data collection completed.")
        self.start_button.setText("Start Data Collection")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 24px;")

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

    def closeEvent(self, event):
        if self.serial_port:
            self.serial_port.close()  # Close the serial port when the application is closed
        event.accept()

# Main function that starts the program
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DataCollectionApp()
    window.show()
    sys.exit(app.exec())