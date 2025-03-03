"""
Name: Auto Impedance PyQt6
Author: Max Chen
Date: September 16, 2024
Description: 
This is a GUI script written with PyQt6 to automate and speed up earphone impedance data collection with the Analog Discovery 2 from Digilent.
Using the API from the DIGILENT WaveForms software, this is a custom script to directly use the existing impedance measuring function in the software.
"""

# Import necessary libraries
import os
import sys
import time
import math
import csv
import threading
from ctypes import *
from dwfconstants import *
import matplotlib.pyplot as plt
# Use QtAgg backend for PyQt6
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QFrame, 
    QGridLayout, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont


class DataCollectionThread(QThread):
    """Thread for data collection to keep the UI responsive"""
    progress_update = pyqtSignal(str)
    plot_update = pyqtSignal(list, list)
    collection_finished = pyqtSignal()
    status_update = pyqtSignal(str)

    def __init__(self, app, folder_name, repetitions):
        super().__init__()
        self.app = app
        self.folder_name = folder_name
        self.repetitions = repetitions
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)

    def run(self):
        # Opens the device, Analog Discovery 2 through the serial port
        hdwf = c_int()
        szerr = create_string_buffer(512)
        self.app.dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

        if hdwf.value == hdwfNone.value:
            self.app.dwf.FDwfGetLastErrorMsg(szerr)
            self.progress_update.emit(f"Failed to open device: {str(szerr.value)}")
            return

        # Impedance Measurement settings
        self.app.dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3))
        sts = c_byte()
        steps = int(self.app.step_entry.text())
        start = int(self.app.start_frequency_entry.text())
        stop = int(self.app.stop_frequency_entry.text())
        reference = int(self.app.reference_entry.text())
        self.status_update.emit("Running")

        # For loop through the specified number of runs
        for run in range(self.repetitions):
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

            # Opens and writes to file
            file_path = os.path.join(self.base_folder, f"{self.folder_name}_Run{run + 1}.csv")
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Frequency (Hz)", "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", "Trace Xs (Ohm)"])

                for i in range(steps):
                    hz = start + i * (stop - start) / (steps - 1)  # linear frequency steps
                    print(f"Run: {run}, Step: {i}, Frequency: {hz} Hz")
                    rgHz[i] = hz
                    self.app.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(hz))  # frequency in Hertz
                    
                    while True:
                        if self.app.dwf.FDwfAnalogImpedanceStatus(hdwf, byref(sts)) == 0:
                            self.app.dwf.FDwfGetLastErrorMsg(szerr)
                            print(str(szerr.value))
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

                    # Write data to CSV
                    writer.writerow([hz, rgTheta[i], rgZ[i], rgRs[i], rgXs[i]])
                    self.progress_update.emit(f"Run {run + 1}, Step {i + 1}/{steps}: Frequency {hz:.2f} Hz, Impedance {rgZ[i]:.2f} Ohms")

            self.plot_update.emit(rgHz, rgZ)
            time.sleep(0.5)

        # Handle end of runs
        self.app.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.app.dwf.FDwfDeviceClose(hdwf)
        self.plot_update.emit(rgHz, rgZ)
        self.progress_update.emit("Data collection completed.")
        self.status_update.emit("Completed")
        self.collection_finished.emit()


class DataCollectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earphones Impedance Data Collector")
        self.showMaximized()  # Set to maximized (windowed fullscreen)
        
        # Load DWF library
        self.load_dwf()
        
        # Setup the UI
        self.setup_ui()

    def setup_ui(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QGridLayout(self.main_widget)
        
        # Top button frame
        self.navi_button_top_frame = QFrame()
        self.navi_button_layout = QHBoxLayout(self.navi_button_top_frame)
        
        # Dropdown for earphone selection
        self.type_label = QLabel("Type:")
        self.type_label.setFont(QFont('Arial', 24))
        self.types = ["A", "B", "C", "D"]
        self.type_combo = QComboBox()
        self.type_combo.setFont(QFont('Arial', 24))
        self.type_combo.addItems(self.types)
        
        # Dropdown for length selection
        self.length_label = QLabel("Length:")
        self.length_label.setFont(QFont('Arial', 24))
        self.lengths = [f"{i}" for i in range(5, 31, 3)] + ["9", "24", "39", "Open", "Blocked"]
        self.length_combo = QComboBox()
        self.length_combo.setFont(QFont('Arial', 24))
        self.length_combo.addItems(self.lengths)
        
        # Label and entry box that takes how many runs are needed
        self.repetitions_label = QLabel('Number of Repetitions: 100')
        self.repetitions_label.setFont(QFont('Arial', 22))
        self.repetitions_entry = QLineEdit()
        self.repetitions_entry.setFont(QFont('Arial', 22))
        self.repetitions_entry.setText("100")
        
        # Start Button which starts the data collection process
        self.start_button = QPushButton("Start Data Collection")
        self.start_button.setFont(QFont('Arial', 24))
        self.start_button.setStyleSheet("background-color: green; color: white;")
        self.start_button.clicked.connect(self.start_data_collection)
        
        # Add widgets to top layout
        self.navi_button_layout.addWidget(self.type_label)
        self.navi_button_layout.addWidget(self.type_combo)
        self.navi_button_layout.addWidget(self.length_label)
        self.navi_button_layout.addWidget(self.length_combo)
        self.navi_button_layout.addWidget(self.repetitions_label)
        self.navi_button_layout.addWidget(self.repetitions_entry)
        self.navi_button_layout.addWidget(self.start_button)
        
        # Left side frame
        self.left_frame = QFrame()
        self.left_layout = QVBoxLayout(self.left_frame)
        
        # Label to indicate the inputted Frequency sweep
        self.frequency_sweep_label = QLabel("Frequency sweep from 20 Hz to 20000 Hz")
        self.frequency_sweep_label.setFont(QFont('Arial', 22))
        
        # Frequency settings grid
        self.settings_grid = QGridLayout()
        
        # Label and Entry for the start frequency
        self.start_frequency_entry_label = QLabel("Start Frequency")
        self.start_frequency_entry_label.setFont(QFont('Arial', 22))
        self.start_frequency_entry = QLineEdit()
        self.start_frequency_entry.setFont(QFont('Arial', 22))
        self.start_frequency_entry.setText("20")
        
        # Label and Entry for the stop frequency
        self.stop_frequency_entry_label = QLabel("Stop Frequency")
        self.stop_frequency_entry_label.setFont(QFont('Arial', 22))
        self.stop_frequency_entry = QLineEdit()
        self.stop_frequency_entry.setFont(QFont('Arial', 22))
        self.stop_frequency_entry.setText("20000")
        
        # Label and Entry for the reference resistor value
        self.reference_entry_label = QLabel("Reference in Ohms")
        self.reference_entry_label.setFont(QFont('Arial', 22))
        self.reference_entry = QLineEdit()
        self.reference_entry.setFont(QFont('Arial', 22))
        self.reference_entry.setText("100")
        
        # Label and Entry for the number of steps per run
        self.step_entry_label = QLabel("STEPS: 501")
        self.step_entry_label.setFont(QFont('Arial', 22))
        self.step_entry = QLineEdit()
        self.step_entry.setFont(QFont('Arial', 22))
        self.step_entry.setText("501")
        
        # Add widgets to settings grid
        self.settings_grid.addWidget(self.start_frequency_entry_label, 0, 0)
        self.settings_grid.addWidget(self.start_frequency_entry, 0, 1)
        self.settings_grid.addWidget(self.stop_frequency_entry_label, 1, 0)
        self.settings_grid.addWidget(self.stop_frequency_entry, 1, 1)
        self.settings_grid.addWidget(self.reference_entry_label, 2, 0)
        self.settings_grid.addWidget(self.reference_entry, 2, 1)
        self.settings_grid.addWidget(self.step_entry_label, 3, 0)
        self.settings_grid.addWidget(self.step_entry, 3, 1)
        
        # Label and Text box to show the progress response
        self.progress_label = QLabel("Status: Ready")
        self.progress_label.setFont(QFont('Arial', 22))
        self.progress_text = QTextEdit()
        self.progress_text.setFont(QFont('Arial', 14))
        self.progress_text.setReadOnly(True)
        
        # Add widgets to left layout
        self.left_layout.addWidget(self.frequency_sweep_label)
        self.left_layout.addLayout(self.settings_grid)
        self.left_layout.addWidget(self.progress_label)
        self.left_layout.addWidget(self.progress_text)
        
        # Create the matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add all elements to the main layout
        self.main_layout.addWidget(self.navi_button_top_frame, 0, 0, 1, 2)
        self.main_layout.addWidget(self.left_frame, 1, 0)
        self.main_layout.addWidget(self.canvas, 1, 1)
        
        # Set column stretch
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 1)
        
    def load_dwf(self):
        if sys.platform.startswith("win"):
            self.dwf = cdll.LoadLibrary("dwf.dll")
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")
    
    def start_data_collection(self):
        # Getting the input from the GUI entry boxes and updating corresponding labels
        selected_type = self.type_combo.currentText()
        selected_length = self.length_combo.currentText()
        repetitions_text = self.repetitions_entry.text()
        
        self.repetitions_label.setText(f"Number of Repetitions: {repetitions_text}")
        self.frequency_sweep_label.setText(f"Frequency sweep from {self.start_frequency_entry.text()} Hz to {self.stop_frequency_entry.text()} Hz")
        
        # Logic to check for repetitions to be always positive
        if not selected_type or not selected_length or not repetitions_text:
            QMessageBox.critical(self, "Error", "Please select type, length, and number of repetitions.")
            return
        
        try:
            repetitions = int(repetitions_text)
            if repetitions <= 0:
                raise ValueError("Number of repetitions must be a positive integer.")
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        
        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # Start data collection thread
        self.start_button.setText("Running")
        self.start_button.setStyleSheet("background-color: red; color: black;")
        
        self.collection_thread = DataCollectionThread(self, folder_name, repetitions)
        self.collection_thread.progress_update.connect(self.update_progress)
        self.collection_thread.plot_update.connect(self.update_plot)
        self.collection_thread.status_update.connect(self.update_status)
        self.collection_thread.collection_finished.connect(self.collection_completed)
        self.collection_thread.start()
    
    def update_progress(self, message):
        self.progress_text.append(message)
        # Scroll to the bottom
        scrollbar = self.progress_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_status(self, status):
        self.progress_label.setText(f"Status: {status}")
    
    def update_plot(self, frequencies, impedance):
        self.ax.clear()
        
        # Ensure valid data for log scaling
        valid_frequencies = [f for f in frequencies if f > 0]
        valid_impedances = [z for z in impedance if z > 0]
        
        if valid_frequencies and valid_impedances:
            self.ax.plot(valid_frequencies, valid_impedances, label='|Z| (Ohms)')
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        else:
            self.ax.plot(frequencies, impedance, label='|Z| (Ohms)')
        
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Impedance (Ohms)')
        self.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.ax.legend()
        self.canvas.draw()
    
    def collection_completed(self):
        QMessageBox.information(self, "Info", "Run Completed!!!")
        self.start_button.setText("Start Data Collection")
        self.start_button.setStyleSheet("background-color: green; color: white;")
    
    def closeEvent(self, event):
        # Handle application close event
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataCollectionApp()
    window.show()
    
    # Handle keyboard interrupts
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Interrupted")
        os._exit(0)
