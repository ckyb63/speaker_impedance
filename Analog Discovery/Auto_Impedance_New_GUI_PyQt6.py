"""
Name: Auto Impedance with PyQt6
Author: Max Chen v25.03.03
Description: 
This is a GUI script written with PyQt6 to automate and speed up earphone impedance data collection with the Analog Discovery 2 from Digilent
Using the API from the DIGILENT WaveForms software, this is a custom script to directly use the existing impedance measuring function in the software.

Prerequisite:
- DIGILENT WaveForms software installed with Python API
"""

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
                           QSizePolicy, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QIcon

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

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
    data_signal = pyqtSignal(float, float)  # Temperature, Humidity
    error_signal = pyqtSignal(str)
    
    def __init__(self, port='COM12', baud_rate=115200):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = True
        self.serial = None
        self.temperature = 25.0
        self.humidity = 50.0
        
    def run(self):
        try:
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=2)
            time.sleep(2)  # Allow time for Arduino to reset
            
            while self.running:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8').strip()
                        parts = line.split(',')
                        if len(parts) >= 2:  # Changed to check for at least 2 parts
                            temp = float(parts[0])
                            humidity = float(parts[1])
                            
                            self.temperature = temp
                            self.humidity = humidity
                            
                            self.data_signal.emit(temp, humidity)
                    except Exception as e:
                        self.error_signal.emit(f"Error parsing Arduino data: {str(e)}")
                time.sleep(0.1)
        except Exception as e:
            self.error_signal.emit(f"Arduino connection error: {str(e)}")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()
    
    def get_latest_data(self):
        return (self.temperature, self.humidity)
                
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
            return
        
        # Impedance Measurement settings from the GUI
        self.app.dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3))
        sts = c_byte()
        steps = int(self.app.step_entry.text())
        start = int(self.app.start_frequency_entry.text())
        stop = int(self.app.stop_frequency_entry.text())
        reference = int(self.app.reference_entry.text())

        # For loop through the specified number of runs
        for run in range(self.repetitions):
            self.progress_signal.emit(f"Starting run {run + 1} of {self.repetitions}")
            
            # Get environmental data
            temp, humidity = self.app.arduino_monitor.get_latest_data()
            self.progress_signal.emit(f"Environmental data: {temp:.1f}°C, {humidity:.1f}%")
            
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
                # Add environmental data to CSV header if enabled
                if self.record_env_data:
                    writer.writerow([f"Temperature (°C): {temp:.2f}", f"Humidity (%): {humidity:.2f}"])
                
                # Update the header row to include environmental data columns if enabled
                if self.record_env_data:
                    writer.writerow(["Frequency (Hz)", "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", "Trace Xs (Ohm)", "Temperature (°C)", "Humidity (%)"])
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

                    # Write data to CSV - include environmental data if enabled
                    if self.record_env_data:
                        writer.writerow([hz, rgTheta[i], rgZ[i], rgRs[i], rgXs[i], temp, humidity])
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
        self.progress_signal.emit("Data collection completed.")
        self.progress_bar_signal.emit(100, 100)  # Ensure progress bar shows 100%
        self.finished_signal.emit()

# Main Data collecting class
class DataCollectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earphones Impedance Data Collector")
        
        # Set the application icon
        ## self.setWindowIcon(QIcon("Analog Discovery/assets/speaker_icon.png"))
        
        # Set fixed initial size with square-ish aspect ratio
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)
        
        # Instead of maximized, start with a square window
        # self.showMaximized()
        
        # Set dark theme
        self.apply_dark_theme()
        
        # Set up the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout is horizontal - control panel on left, plot on right
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Initialize Arduino connection
        self.arduino_monitor = ArduinoMonitor(port='COM12')
        self.arduino_monitor.data_signal.connect(self.update_environmental_data)
        self.arduino_monitor.error_signal.connect(self.handle_arduino_error)
        self.arduino_monitor.start()
        
        # Create control panel group
        self.create_control_panel()
        
        # Create plot panel
        self.create_plot_panel()
        
        # Load DWF library
        self.load_dwf()

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
        self.control_group.setMaximumWidth(450)
        
        self.input_layout = QVBoxLayout(self.control_group)
        self.input_layout.setSpacing(5)
        self.input_layout.setContentsMargins(10, 15, 10, 10)

        # Start Button with improved styling
        self.start_button = QPushButton("Start Data Collection")
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
        self.start_button.clicked.connect(self.start_data_collection)
        self.input_layout.addWidget(self.start_button)

        # Configuration section with reduced spacing
        config_header = QLabel("Configuration")
        config_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.input_layout.addWidget(config_header)

        # Dropdowns with improved styling and reduced height
        dropdown_style = """
            QComboBox { 
                padding: 3px;
                max-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """

        self.types = ["A", "B", "C", "D"]
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.types)
        self.type_combo.setStyleSheet(dropdown_style)
        self.input_layout.addWidget(QLabel("Earphone Type:"))
        self.input_layout.addWidget(self.type_combo)

        self.lengths = [f"{i}" for i in range(5, 31, 3)] + ["9", "24", "39", "Open", "Blocked"]
        self.length_combo = QComboBox()
        self.length_combo.addItems(self.lengths)
        self.length_combo.setStyleSheet(dropdown_style)
        self.input_layout.addWidget(QLabel("Length (mm):"))
        self.input_layout.addWidget(self.length_combo)

        # Environmental readings section
        env_header = QLabel("Environmental Readings")
        env_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.input_layout.addWidget(env_header)
        
        # Create a grid for environmental readings
        env_grid = QGridLayout()
        env_grid.setSpacing(5)
        
        # Temperature reading
        temp_label = QLabel("Temperature:")
        self.temp_value = QLabel("25.0 °C")
        self.temp_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        env_grid.addWidget(temp_label, 0, 0)
        env_grid.addWidget(self.temp_value, 0, 1)
        
        # Humidity reading
        humidity_label = QLabel("Humidity:")
        self.humidity_value = QLabel("50.0 %")
        self.humidity_value.setStyleSheet("color: #58a6ff; font-weight: bold;")
        env_grid.addWidget(humidity_label, 1, 0)
        env_grid.addWidget(self.humidity_value, 1, 1)
        
        # Add checkbox for recording environmental data
        self.env_data_checkbox = QCheckBox("Record Environmental Data")
        self.env_data_checkbox.setChecked(True)  # Enable by default
        self.env_data_checkbox.setStyleSheet("color: #ffffff;")
        env_grid.addWidget(self.env_data_checkbox, 2, 0, 1, 2)  # span 2 columns
        
        self.input_layout.addLayout(env_grid)

        # Advanced settings section with reduced spacing
        advanced_header = QLabel("Advanced Settings")
        advanced_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.input_layout.addWidget(advanced_header)

        # Entry fields with improved styling and reduced height
        entry_style = """
            QLineEdit {
                padding: 3px;
                max-height: 24px;
                margin: 0px;
            }
        """

        # Repetitions (keep this one vertical as it's the main control)
        self.repetitions_entry = QLineEdit("100")
        self.repetitions_entry.setStyleSheet(entry_style)
        self.input_layout.addWidget(QLabel("Number of Repetitions:"))
        self.input_layout.addWidget(self.repetitions_entry)

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
        stop_freq_layout.addWidget(QLabel("Stop Frequency (Hz):"))
        stop_freq_layout.addWidget(self.stop_frequency_entry)
        freq_layout.addWidget(stop_freq_widget)
        
        self.input_layout.addLayout(freq_layout)
        
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
        steps_layout.addWidget(QLabel("STEPS:"))
        steps_layout.addWidget(self.step_entry)
        ref_steps_layout.addWidget(steps_widget)
        
        self.input_layout.addLayout(ref_steps_layout)
        
        # Status section with reduced height
        status_header = QLabel("Status")
        status_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.input_layout.addWidget(status_header)

        # Progress label
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                min-height: 20px;
            }
        """)
        self.input_layout.addWidget(self.progress_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
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
        self.input_layout.addWidget(self.progress_bar)
        
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

        # Add control group to the main layout with 1:2 ratio with plot
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
        self.plot_group.setMinimumWidth(600)
        
        plot_layout = QVBoxLayout(self.plot_group)
        plot_layout.setSpacing(5)
        plot_layout.setContentsMargins(10, 15, 10, 10)

        # Create the matplotlib canvas with dark theme - using square dimensions
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        self.canvas.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.canvas.ax.set_xlabel('Frequency (Hz)')
        self.canvas.ax.set_ylabel('Impedance (Ohms)')
        self.canvas.ax.set_xscale('log')
        self.canvas.ax.set_yscale('log')
        
        plot_layout.addWidget(self.canvas)
        
        # Add plot group to the main layout - now use stretch factor 3 to give more space
        self.main_layout.addWidget(self.plot_group, stretch=3)

    def update_environmental_data(self, temperature, humidity):
        """Update the environmental data labels with values from Arduino"""
        self.temp_value.setText(f"{temperature:.1f} °C")
        self.humidity_value.setText(f"{humidity:.1f} %")
        
    def handle_arduino_error(self, error_message):
        """Handle Arduino connection errors"""
        self.update_progress(f"Arduino Error: {error_message}")
        # Update the environmental labels to show error state
        self.temp_value.setText("Error")
        self.humidity_value.setText("Error")
        self.temp_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.humidity_value.setStyleSheet("color: #e74c3c; font-weight: bold;")

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
        
        # Also update the status label with the latest message
        self.progress_label.setText(message)

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
        # For logarithmic scales, 'equal' doesn't work directly, so we need to handle differently
        # We'll set the aspect ratio to be 'auto' but ensure the box is square
        self.canvas.ax.set_box_aspect(1.0)
        
        # Update the canvas
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def start_data_collection(self):
        """Start the data collection process in a separate thread"""
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
            return
        
        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        
        # If environmental data recording is enabled, append temperature and humidity to folder name
        if self.env_data_checkbox.isChecked():
            # Get current environmental data
            temp, humidity = self.arduino_monitor.get_latest_data()
            folder_name = f"{folder_name}_T{temp:.1f}C_H{humidity:.1f}pct"
            
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
            self.update_progress(f"Created data folder: {folder_name}")
        
        # Update UI
        self.progress_label.setText("Status: Running")
        self.start_button.setText("Running...")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f0883e;
                color: white;
                font-size: 18px;
                padding: 8px;
                border-radius: 4px;
                border: none;
            }
        """)
        self.start_button.setEnabled(False)
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Start data collection in a worker thread
        self.worker = DataCollectionWorker(self, folder_name, repetitions, self.env_data_checkbox.isChecked())
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.plot_signal.connect(self.update_plot)
        self.worker.progress_bar_signal.connect(self.update_progress_bar)
        self.worker.finished_signal.connect(self.collection_finished)
        self.worker.start()
        
        self.update_progress(f"Starting data collection for {selected_type}_{selected_length} with {repetitions} repetitions")

    def collection_finished(self):
        """Update UI after data collection is complete"""
        self.progress_label.setText("Status: Complete")
        self.start_button.setText("Start Data Collection")
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
        self.start_button.setEnabled(True)
        
        # Show the export dataset button if environmental data recording was enabled
        if hasattr(self, 'worker') and self.worker.record_env_data:
            if not hasattr(self, 'export_dataset_button'):
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
                    QPushButton:hover {
                        background-color: #3b3b9c;
                    }
                """)
                self.export_dataset_button.clicked.connect(self.export_ml_dataset)
                self.input_layout.addWidget(self.export_dataset_button)
            self.export_dataset_button.setVisible(True)
        
        # Show completion message
        QMessageBox.information(self, "Data Collection Complete", 
                              "All measurement runs have been completed successfully!\n\n"
                              "The data has been saved to CSV files in the specified folder.")
        
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
                env_data_in_name = False
                temperature_str = "N/A"
                humidity_str = "N/A"
                
                if len(folder_parts) >= 4 and folder_parts[2].startswith('T') and folder_parts[3].startswith('H'):
                    env_data_in_name = True
                    # Extract temperature and humidity from folder name
                    temperature_str = folder_parts[2][1:-1]  # Remove 'T' prefix and 'C' suffix
                    humidity_str = folder_parts[3][1:-3]     # Remove 'H' prefix and 'pct' suffix
            else:
                earphone_type = "unknown"
                earphone_length = "unknown"
                
            # Create the output file name based on type and length
            output_file = os.path.join(ml_folder, f"{earphone_type}_{earphone_length}_All.csv")
            
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
                writer.writerow([
                    "Type", "Length", "Run", "Frequency (Hz)", 
                    "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", 
                    "Trace Xs (Ohm)", "Temperature (°C)", "Humidity (%)"
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
                        has_env_data = len(rows[header_rows]) > 5
                        
                        # Process data rows
                        for row in rows[header_rows:]:
                            if not row:  # Skip empty rows
                                continue
                                
                            # Extract data
                            if has_env_data:
                                freq, theta, z, rs, xs, temp, humidity = row
                            else:
                                freq, theta, z, rs, xs = row
                                temp = "N/A"
                                humidity = "N/A"
                                
                            # Write consolidated row
                            writer.writerow([
                                earphone_type, earphone_length, run_number,
                                freq, theta, z, rs, xs, temp, humidity
                            ])
            
            self.update_progress(f"ML training dataset exported to: {earphone_type}_{earphone_length}_All.csv")
            
            # Ask if user wants to open the folder
            reply = QMessageBox.question(
                self, 
                'Dataset Exported', 
                f'Dataset was exported as:\n{earphone_type}_{earphone_length}_All.csv\n\nWould you like to open the folder?',
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
        if hasattr(self, 'arduino_monitor'):
            self.arduino_monitor.stop()
            
        if hasattr(self, 'worker') and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Exit', 
                                      'Data collection is still in progress. Are you sure you want to exit?',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                      QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.terminate()  # Force terminate the worker thread
                event.accept()
                sys.exit(0)
            else:
                event.ignore()
        else:
            reply = QMessageBox.question(self, 'Exit', 
                                      'Are you sure you want to exit the application?',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                      QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
                sys.exit(0)
            else:
                event.ignore()

# Main function that starts the program
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = DataCollectionApp()
    window.show()
    
    sys.exit(app.exec()) 