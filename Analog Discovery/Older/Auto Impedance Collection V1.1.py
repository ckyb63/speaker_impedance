"""
Name: Auto Impedance V1.1
Author: Max Chen
Date: September 16, 2024
Description: 
This is a GUI script written with tkinter to automate and speed up earphone impedance data collection with the Analog Discovery 2 from Digilent
Using the API from the DIGILENT WaveForms software, this is a custom script to directly use the exisiting impedance measuring function in the software.
"""

# Import necessary libraries.
import os
import sys
import tkinter as tk
from tkinter import messagebox
from ctypes import *
from dwfconstants import *
import time
import math
import matplotlib.pyplot as plt
import csv
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  

# Main Data collecting class
class DataCollectionApp:

    #Self initial function which contains the GUI creation and layout code. 
    def __init__(self, root):
        self.root = root
        self.root.title("Earphones Impedance Data Collector")
        self.root.state('zoomed') # Set to zooooooomed (windowed fullscreen)

        # Main frame of the window and its configurations 
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Increased padding
        
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)  # Second column for plot canvas
        
        # Frame for for the top row of buttons
        self.navi_button_top_frame = tk.Frame(self.main_frame, height=50)
        self.navi_button_top_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky='ew')  # Span both columns
        for i in range(5):
            self.navi_button_top_frame.grid_columnconfigure(i, weight=1)

        # Dropdown for earphone selection
        self.types = ["A", "B", "C", "D"]
        self.type_combo = tk.StringVar(value=self.types[0])
        self.type_menu_mode = tk.OptionMenu(self.navi_button_top_frame, self.type_combo, *self.types)
        self.type_menu_mode.config(width=20, font=('Arial', 24))
        self.type_menu_mode.grid(row=0, column=0, padx=(0, 10), sticky='ew')
        # Configure the font size of the dropdown menu items for mode selection
        menu_mode = self.type_menu_mode.nametowidget(self.type_menu_mode.menuname)
        menu_mode.config(font=('Arial', 24))

        # Dropdown for length selection
        self.lengths = [f"{i}" for i in range(5, 31, 3)] + [9, 24, 39] + ["Open", "Blocked"]
        self.length_combo = tk.StringVar(value=self.lengths[0])
        self.type_menu = tk.OptionMenu(self.navi_button_top_frame, self.length_combo, *self.lengths)
        self.type_menu.config(width=20, font=('Arial', 24))
        self.type_menu.grid(row=0, column=1, padx=(0, 10), sticky='ew')
        # Configure the font size of the dropdown menu items
        menu = self.type_menu.nametowidget(self.type_menu.menuname)
        menu.config(font=('Arial', 24))

        # Label and entry box that takes how many runs are needed 
        self.reptitions_label = tk.Label(self.navi_button_top_frame, text='Number of Repetitions: 000', font=('Arial', 22))
        self.reptitions_label.grid(row=0, column=2, padx=(5, 10), pady=(5, 10), sticky='w')
        self.repetitions_entry = tk.Entry(self.navi_button_top_frame, font=('Arial', 22), width=10)
        self.repetitions_entry.grid(row=0, column=3, padx=(5, 10), sticky='ew')
        self.repetitions_entry.insert(0, "100")

        # Start Button which starts the data collection process
        self.Start_button = tk.Button(self.navi_button_top_frame, text="Start Data Collection", command=self.start_data_collection, font=('Arial', 24), bg='green', fg='white')
        self.Start_button.grid(row=0, column=4, padx=10, sticky='ew') 

        # Left side of the screen and config
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        for i in range(6):
            self.left_frame.grid_rowconfigure(i, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Label to indicate the inputted Frequency sweep 
        self.frequency_sweep_label = tk.Label(self.left_frame, text="Frequency sweep from 0 Hz to 0 Hz", font=('Arial', 22))
        self.frequency_sweep_label.grid(row=0, column=0, columnspan=2, padx=(5, 10), pady=(5, 10), sticky='w')

        # Label and Entry for the start frequency
        self.start_frequency_entry_label = tk.Label(self.left_frame, text="Start Frequency", font=('Arial', 22))
        self.start_frequency_entry_label.grid(row=1, column=0, padx=(5, 10), pady=(5, 10), sticky='w')
        self.start_frequency_entry = tk.Entry(self.left_frame, font=('Arial', 22), width=10)
        self.start_frequency_entry.grid(row=1, column=1, padx=(5, 10), sticky='ew')
        self.start_frequency_entry.insert(0, "20")

        # Label and Entry for the stop frequency //Note: There is no logic check if input is wrong
        self.stop_frequency_entry_label = tk.Label(self.left_frame, text="Stop Frequency", font=('Arial', 22))
        self.stop_frequency_entry_label.grid(row=2, column=0, padx=(5, 10), pady=(5, 10), sticky='w')
        self.stop_frequency_entry = tk.Entry(self.left_frame, font=('Arial', 22), width=10)
        self.stop_frequency_entry.grid(row=2, column=1, padx=(5, 10), sticky='ew')
        self.stop_frequency_entry.insert(0, "20000")

        # Label and Entry for the reference resistor value, this should always be 100 and doesn't change unless hardware. 
        self.reference_entry_label = tk.Label(self.left_frame, text="Reference in Ohms", font=('Arial', 22))
        self.reference_entry_label.grid(row=3, column=0, padx=(5, 10), pady=(5, 10), sticky='w')
        self.reference_entry = tk.Entry(self.left_frame, font=('Arial', 22), width=10)
        self.reference_entry.grid(row=3, column=1, padx=(5, 10), sticky='ew')
        self.reference_entry.insert(0, "100")

        # Label and Entry for the number of steps per run
        self.step_entry_label = tk.Label(self.left_frame, text="STEPS: 501", font=('Arial', 22))
        self.step_entry_label.grid(row=4, column=0, padx=(5, 10), pady=(5, 10), sticky='w')
        self.step_entry = tk.Entry(self.left_frame, font=('Arial', 22), width=10)
        self.step_entry.grid(row=4, column=1, padx=(5, 10), sticky='ew')
        self.step_entry.insert(0, "501")

        # Label and Text box to show the progress response, 
        self.progress_label = tk.Label(self.left_frame, text="Status:", font=('Arial', 22),)
        self.progress_label.grid(row=5, column=0, columnspan=2, padx=5, sticky='ew')
        self.progress_text = tk.Text(self.left_frame, height=15, width=50,font=('Arial', 14), wrap=tk.WORD)
        self.progress_text.grid(row=6, column=0, columnspan=2, padx=5, sticky='sew')

        # Create initial plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=1, sticky='nsew', pady=20)

        # Load DWF library
        self.load_dwf()

        # After popup is finished, the program will close all windows. 
        def close_window():
            self.root.destroy()
            self.root.quit()
            os._exit(0)

        # Bind the window close event to the on_closing function
        root.protocol("WM_DELETE_WINDOW", close_window)

    # Function loads the dwf library
    def load_dwf(self):
        if sys.platform.startswith("win"):
            self.dwf = cdll.LoadLibrary("dwf.dll")
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")

    # Function Starts the data collection process 
    def start_data_collection(self):
        # Getting the input from the GUI entry boxes and updating corresponding labels
        selected_type = self.type_combo.get()
        selected_length = self.length_combo.get()
        repetitions = self.repetitions_entry.get()
        self.reptitions_label.config(text=f"Number of Repetitions: {self.repetitions_entry.get()}")
        self.frequency_sweep_label.config(text=f"Frequency sweep from {self.start_frequency_entry.get()} Hz to {self.stop_frequency_entry.get()} Hz")

        # Logic to check for repetitions to be alwasy positive
        if not selected_type or not selected_length or not repetitions:
            messagebox.showerror("Error", "Please select type, length, and number of repetitions.")
            return
        try:
            repetitions = int(repetitions)
            if repetitions <= 0:
                raise ValueError("Number of repetitions must be a positive integer.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Collected_Data", folder_name)
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        
        # Start data collection in a new thread to keep the GUI responsive
        thread = threading.Thread(target=self.collect_data, args=(folder_name, repetitions))
        thread.start()

    # Function that actaully collects the data
    def collect_data(self, folder_name, repetitions):

        # Opens the device, Analog Discovery 2 through the serial port.
        hdwf = c_int()
        szerr = create_string_buffer(512)
        self.dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

        if hdwf.value == hdwfNone.value:
            self.dwf.FDwfGetLastErrorMsg(szerr)
            self.update_progress(f"Failed to open device: {str(szerr.value)}")
            return
        
        # Impedance Measurement settings getting the numbers from the GUI as well as update GUI corresponding Labels.
        self.dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3))
        sts = c_byte()
        #steps = 501
        #start = 20
        #stop = 20000z
        #reference = 100
        steps = int(self.step_entry.get())
        start = int(self.start_frequency_entry.get())
        stop = int(self.stop_frequency_entry.get())
        reference = int(self.reference_entry.get())
        self.progress_label.config(text="Status: Running")
        self.Start_button.config(text="Running", font=('Arial', 24), bg='red', fg='black') # Start button changes depending on status

        # For loop through the specified number of runs
        for run in range(repetitions):

            # Configure the settings for impedance measurements
            self.dwf.FDwfAnalogImpedanceReset(hdwf)
            self.dwf.FDwfAnalogImpedanceModeSet(hdwf, c_int(0))
            self.dwf.FDwfAnalogImpedanceReferenceSet(hdwf, c_double(reference))
            self.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(start))
            self.dwf.FDwfAnalogImpedanceAmplitudeSet(hdwf, c_double(0.5))
            self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(1))
            time.sleep(0.5)

            # Create Values for data we are interested in
            rgHz = [0.0]*steps
            rgTheta = [0.0]*steps
            rgZ = [0.0]*steps
            rgRs = [0.0]*steps
            rgXs = [0.0]*steps

            # Opens and writes to file based on the file structure explained seperately.. 
            file_path = os.path.join(self.base_folder, f"{folder_name}_Run{run + 1}.csv")
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Frequency (Hz)", "Trace θ (deg)", "Trace |Z| (Ohm)", "Trace Rs (Ohm)", "Trace Xs (Ohm)"])

                for i in range(steps):
                    hz = start + i * (stop - start) / (steps - 1) # linear frequency steps
                    print(f"Run: {run}, Step: {i}, Frequency: {hz} Hz") # Terminal and GUI status text progress
                    rgHz[i] = hz
                    self.dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(hz)) # frequency in Hertz
                    # if settle time is required for the DUT, wait and restart the acquisition
                    #time.sleep(0.01) 
                    #self.dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))
                    while True:
                        if self.dwf.FDwfAnalogImpedanceStatus(hdwf, byref(sts)) == 0:
                            self.dwf.FDwfGetLastErrorMsg(szerr)
                            print(str(szerr.value))
                            quit()
                        if sts.value == 2:
                            break
                    resistance = c_double()
                    reactance = c_double()
                    self.dwf.FDwfAnalogImpedanceStatusMeasure(hdwf, DwfAnalogImpedanceResistance, byref(resistance))
                    self.dwf.FDwfAnalogImpedanceStatusMeasure(hdwf, DwfAnalogImpedanceReactance, byref(reactance))
                    rgRs[i] = resistance.value
                    rgXs[i] = reactance.value
                    rgZ[i] = (resistance.value**2 + reactance.value**2)**0.5
                    rgTheta[i] = math.degrees(math.atan2(reactance.value, resistance.value))

                    # Write data to CSV
                    writer.writerow([hz, rgTheta[i], rgZ[i], rgRs[i], rgXs[i]])
                    self.update_progress(f"Run {run + 1}, Step {i + 1}/{steps}: Frequency {hz:.2f} Hz, Impedance {rgZ[i]:.2f} Ohms")

            self.update_plot(rgHz, rgZ)
            time.sleep(0.5)

        # Code to hanle run 100, as well as update necessary GUI elements
        self.dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0))
        self.dwf.FDwfDeviceClose(hdwf)
        self.update_plot(rgHz, rgZ)
        self.update_progress("Data collection completed.")
        self.progress_label.config(text="Status: Completed")
        messagebox.showinfo("Info", "Run Completed!!!")
        self.Start_button.config(text="Start Data Collection", font=('Arial', 24), bg='green', fg='white')

    # Update progress function what adds the current printing message to the text-box GUI element.
    def update_progress(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)

    # Updates the plot which provides visulaiztion of the collected data for the user.
    def update_plot(self, frequencies, impedance):
        self.ax.clear()
        time.sleep(0.1)
        
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
        
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Impedance (Ohms)')
        self.ax.set_title('Impedance Magnitude |Z| vs Frequency')
        self.ax.legend()
        self.canvas.draw()

# "Main" function that starts the program
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()
    root.lift()

# Program termination from keyboard interrupts in the terminal
try:
    while True: time.sleep(0.1)
except KeyboardInterrupt:
    print("Interrupted")
    os._exit(0)    