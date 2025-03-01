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
from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import serial

class DataCollectionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earphones Impedance Data Collector")
        self.setGeometry(100, 100, 1200, 600)
        
        # Create and set the application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "speaker_icon.png")
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
                max-height: 100px;
            }
        """)

        # Initialize serial communication
        self.serial_port = None
        try:
            self.serial_port = serial.Serial('COM8', 115200, timeout=1)
        except serial.SerialException:
            pass

        # Main layout
        self.main_layout = QtWidgets.QHBoxLayout(self)

        # Create a group box for the control panel
        self.control_group = QtWidgets.QGroupBox("Control Panel")
        self.control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        self.input_layout = QtWidgets.QVBoxLayout(self.control_group)
        self.input_layout.setSpacing(5)
        self.input_layout.setContentsMargins(5, 5, 5, 5)

        # Start Button with improved styling
        self.start_button = QtWidgets.QPushButton("Start Data Collection")
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
        config_header = QtWidgets.QLabel("Configuration")
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
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(self.types)
        self.type_combo.setStyleSheet(dropdown_style)
        self.input_layout.addWidget(QtWidgets.QLabel("Earphone Type:"))
        self.input_layout.addWidget(self.type_combo)

        self.lengths = [str(i) for i in range(5, 31, 3)] + [str(i) for i in [9, 24, 39]] + ["Open", "Blocked"]
        self.length_combo = QtWidgets.QComboBox()
        self.length_combo.addItems(self.lengths)
        self.length_combo.setStyleSheet(dropdown_style)
        self.input_layout.addWidget(QtWidgets.QLabel("Length (mm):"))
        self.input_layout.addWidget(self.length_combo)

        # Advanced settings section with reduced spacing
        advanced_header = QtWidgets.QLabel("Advanced Settings")
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
        self.repetitions_entry = QtWidgets.QLineEdit("100")
        self.repetitions_entry.setStyleSheet(entry_style)
        self.input_layout.addWidget(QtWidgets.QLabel("Number of Repetitions:"))
        self.input_layout.addWidget(self.repetitions_entry)

        # Create horizontal layouts for paired settings
        freq_layout = QtWidgets.QHBoxLayout()
        freq_layout.setSpacing(10)
        
        # Start Frequency
        start_freq_widget = QtWidgets.QWidget()
        start_freq_layout = QtWidgets.QVBoxLayout(start_freq_widget)
        start_freq_layout.setSpacing(3)
        start_freq_layout.setContentsMargins(0, 0, 0, 0)
        self.start_frequency_entry = QtWidgets.QLineEdit("20")
        self.start_frequency_entry.setStyleSheet(entry_style)
        start_freq_layout.addWidget(QtWidgets.QLabel("Start Frequency (Hz):"))
        start_freq_layout.addWidget(self.start_frequency_entry)
        freq_layout.addWidget(start_freq_widget)
        
        # Stop Frequency
        stop_freq_widget = QtWidgets.QWidget()
        stop_freq_layout = QtWidgets.QVBoxLayout(stop_freq_widget)
        stop_freq_layout.setSpacing(3)
        stop_freq_layout.setContentsMargins(0, 0, 0, 0)
        self.stop_frequency_entry = QtWidgets.QLineEdit("20000")
        self.stop_frequency_entry.setStyleSheet(entry_style)
        stop_freq_layout.addWidget(QtWidgets.QLabel("Stop Frequency (Hz):"))
        stop_freq_layout.addWidget(self.stop_frequency_entry)
        freq_layout.addWidget(stop_freq_widget)
        
        self.input_layout.addLayout(freq_layout)
        
        # Reference and Steps in another horizontal layout
        ref_steps_layout = QtWidgets.QHBoxLayout()
        ref_steps_layout.setSpacing(10)
        
        # Reference
        ref_widget = QtWidgets.QWidget()
        ref_layout = QtWidgets.QVBoxLayout(ref_widget)
        ref_layout.setSpacing(3)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_entry = QtWidgets.QLineEdit("100")
        self.reference_entry.setStyleSheet(entry_style)
        ref_layout.addWidget(QtWidgets.QLabel("Reference in Ohms:"))
        ref_layout.addWidget(self.reference_entry)
        ref_steps_layout.addWidget(ref_widget)
        
        # Steps
        steps_widget = QtWidgets.QWidget()
        steps_layout = QtWidgets.QVBoxLayout(steps_widget)
        steps_layout.setSpacing(3)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        self.step_entry = QtWidgets.QLineEdit("501")
        self.step_entry.setStyleSheet(entry_style)
        steps_layout.addWidget(QtWidgets.QLabel("STEPS:"))
        steps_layout.addWidget(self.step_entry)
        ref_steps_layout.addWidget(steps_widget)
        
        self.input_layout.addLayout(ref_steps_layout)

        # Environmental readings section with reduced spacing
        readings_header = QtWidgets.QLabel("Environmental Readings")
        readings_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.input_layout.addWidget(readings_header)

        # Create a more compact grid layout for readings
        readings_grid = QtWidgets.QGridLayout()
        readings_grid.setSpacing(5)
        readings_grid.setContentsMargins(0, 0, 0, 0)

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

        self.input_layout.addLayout(readings_grid)

        # Status section with reduced height
        status_header = QtWidgets.QLabel("Status")
        status_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #58a6ff; margin-top: 5px;")
        self.input_layout.addWidget(status_header)

        self.progress_text = QtWidgets.QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(80)
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

        # Create a group box for the plot
        self.plot_group = QtWidgets.QGroupBox("Impedance Plot")
        self.plot_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                color: #ffffff;
            }
        """)
        plot_layout = QtWidgets.QVBoxLayout(self.plot_group)
        plot_layout.setSpacing(5)
        plot_layout.setContentsMargins(5, 5, 5, 5)

        # Create and style the plot
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        plot_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.plot_group, stretch=2)

        # Set the title of the plot
        self.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Impedance (Ohms)')

        # Load DWF library
        self.load_dwf()

        # Update progress based on Arduino connection status
        if self.serial_port:
            self.update_progress("Arduino connected.")
        else:
            self.update_progress("Warning: Arduino not connected. Data collection will not include temperature, humidity, and pressure.")

        # Timer for reading Arduino data
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.read_arduino_data)

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

        # Update button appearance for running state
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f85149;
                color: white;
                font-size: 18px;
                padding: 8px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ff6b64;
            }
        """)
        self.start_button.setText("Running...")

        # Start the timer to read Arduino data every second
        self.timer.start(1000)

    def read_arduino_data(self):
        if self.serial_port and self.serial_port.in_waiting > 0:
            line = self.serial_port.readline().decode('utf-8').rstrip()
            try:
                temperature, humidity, pressure = map(float, line.split(','))
                self.temperature_value.setText(f"{temperature:.2f}")
                self.humidity_value.setText(f"{humidity:.2f}")
                self.pressure_value.setText(f"{pressure:.2f}")
                return temperature, humidity, pressure
            except ValueError:
                print("Invalid data from Arduino: ", line)

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

        for run in range(repetitions):
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
                    self.update_progress(f"Run {run + 1}, Step {i + 1}/{steps}: Frequency {hz:.2f} Hz, Impedance {rgZ[i]:.2f} Ohms")

            self.update_plot(rgHz, rgZ)
            time.sleep(0.5)

        self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.dwf.FDwfDeviceClose(hdwf)

        # Stop the timer after data collection is complete
        self.timer.stop()
        self.update_progress("Data collection completed.")
        
        # Reset button appearance
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

    def update_progress(self, message):
        self.progress_text.append(message)
        self.progress_text.verticalScrollBar().setValue(self.progress_text.verticalScrollBar().maximum())

    def update_plot(self, frequencies, impedance):
        self.ax.clear()
        self.ax.plot(frequencies, impedance, label='|Z| (Ohms)', color='#58a6ff', linewidth=2)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Impedance (Ohms)')
        self.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.legend()
        
        # Ensure dark theme is maintained
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        self.canvas.draw()

    def closeEvent(self, event):
        if self.serial_port:
            self.serial_port.close()
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
    window = DataCollectionApp()
    window.show()
    sys.exit(app.exec())