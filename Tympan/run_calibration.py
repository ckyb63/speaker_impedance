#
# run_calibration.py
# 
# Chip Audette, Max Chen, OpenAudio, Oct 2024
# MIT License 
#
# This script communicates with the Tympan over the USB Serial link.
# The purpose is to run the stepped-tone calibration and to retrieve
# the results
#
# I'm using this as an example: https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756
#

import serial  #pip install pyserial
import time 
import matplotlib.pyplot as plt
import codecs
import numpy as np

import os
import sys
import time
import tkinter as tk
from tkinter import messagebox
import math
import csv
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ##################### Define functions
def clearSerialBuffer():
    foo = getReply(False,1.0)

def printReceivedLine(all_lines):
    #if (len(all_lines) > 0) and (all_lines[-1]=='\n'):
    #    all_lines = all_lines[:-1] #strip off trailing \n
    #if (len(all_lines) > 0) and (all_lines[-1]=='\r'):
    #    all_lines = all_lines[:-1] #strip off trailing \r
    print(all_lines,end='')

def getReply(print_as_received=True,wait_period_sec=0.5):
    all_lines = ''
    new_readline = codecs.decode(serial_with_tympan.readline(),encoding='utf-8')
    all_lines += new_readline
    last_reply_time  = time.time()
    #while (len(new_readline) > 0):
    while (time.time() < (last_reply_time + wait_period_sec)):
        new_readline = codecs.decode(serial_with_tympan.readline(),encoding='utf-8')
        if len(new_readline) > 0:
            last_reply_time = time.time()
        all_lines += new_readline
        if print_as_received:
            printReceivedLine(new_readline)
    return all_lines

def sendCharacterAndGetResponse(send_character, print_as_received = True, wait_period_sec = 0.5):
    serial_with_tympan.write(bytes(send_character + '\n', 'utf-8'))  #send an 'h' to the Tympan
    time.sleep(0.05)                             #wait a bit to enable a response 
    all_lines = getReply(print_as_received, wait_period_sec)
    return all_lines

def parseTestDataString(all_lines):
    all_vals = []
    list_strings = all_lines.splitlines()  # convert long string (with newlines) into list of strings
    test_string = 'V):'
    
    for line in list_strings:
        ind = line.find(test_string)
        if ind != -1:
            ind += len(test_string)
            line = line[ind:].strip()  # Trim any leading or trailing whitespace
            vals = np.fromstring(line, dtype=float, sep=',')
            
            # Check if we have exactly 4 values: step, tone frequency, left dBFS, and right dBFS
            if len(vals) == 4:
                all_vals.append(vals)
            else:
                print(f"Warning: Skipping line with unexpected number of values: {line}")

    if len(all_vals) == 0:
        return np.array([]), np.array([]), np.array([])  # return empty arrays if no valid data
    
    all_vals = np.array(all_vals)  # Now, all rows should have consistent length
    
    # Split into step, freq_Hz, and input_v (Left and Right)
    test_id = all_vals[:, 0]  # First column: step number
    freq_Hz = all_vals[:, 1]  # Second column: tone frequency
    input_v = all_vals[:, 2:]  # Remaining columns: Left and Right dBFS
    
    return test_id, freq_Hz, input_v

    

# #################33 Here is the Main

#specify the COM port of the Tympan
my_com_port = 'COM4'  #Look in the Arduino IDE! 

# create a serial instance for communicating to our device
print("Opening serial port...make sure the Serial Monitor is closed in Arduino IDE")
serial_with_tympan = serial.Serial(port=my_com_port, baudrate=115200, timeout=.5)

# mute the system to stop any curren test
all_lines = sendCharacterAndGetResponse('m',print_as_received=False, wait_period_sec=0.5)

# reset the test parameters
all_lines = sendCharacterAndGetResponse('q')

# ask the Tympan for the help menu and read the response
all_lines = sendCharacterAndGetResponse('h')

wait_period_sec = 1.2  # default value for slow (1 second)
if 1:
    # speed up the test by shorting from the default 0.5sec/step to 0.2 sec/step the test parameters
    all_lines = sendCharacterAndGetResponse('DDD')
    wait_period_sec = 0.5  #faster value for faster (0.2 sec) test tones

# Run iterator and label
run = 0
limit = 50
selected_type = "B"
selected_length = "14mm"

for i in range(limit): #For loop 50 times
    run += 1 #Increment Run

    # command the test to start
    all_lines = sendCharacterAndGetResponse('T',wait_period_sec=wait_period_sec)

    # get all of the results
    all_lines = sendCharacterAndGetResponse('v')

    # parse the data
    test_id, freq_Hz, input_v = parseTestDataString(all_lines)

    if input_v.size == 0:
        print("Error: No valid data was parsed from the response.")
    # Check if we have two channels for Left and Right data (voltages)
    elif input_v.shape[1] >= 2:  # ensure at least two columns exist
        left_values = input_v[:, 0]  # Left channel voltage
        right_values = input_v[:, 1]  # Right channel voltage

        # Calculate impedance using the formula Z = (V2 - V1) * 100 / V2
        impedance = ((left_values - right_values) * 100) / left_values
        print(impedance)

        # Plot Voltage for Left and Right
        plt.figure(figsize=(10, 6))
        #plt.subplot(1, 2, 1)
        #plt.semilogx(freq_Hz, left_values, label='Left Channel', linewidth=2)
        #plt.semilogx(freq_Hz, right_values, label='Right Channel', linewidth=2)
        #plt.xlabel('Frequency (Hz)')
        #plt.ylabel('Voltage (V)')
        #plt.legend()
        #plt.title('Voltage (Left and Right Channels)')

        # Plot Impedance
        #plt.subplot(1, 2, 2)
        #plt.semilogx(freq_Hz, impedance, label='Impedance', linewidth=2, color='orange')
        #plt.xlabel('Frequency (Hz)')
        #plt.ylabel('Impedance')
        #plt.legend()
        #plt.title('Impedance between Left and Right Channels')
        #plt.tight_layout()
        #plt.show()

        # Folder selection and creation
        folder_name = f"{selected_type}_{selected_length}"
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tympan_collected_Data", folder_name)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        # Opens and writes to file based on the file structure explained seperately.. 
        file_path = os.path.join(base_folder, f"{folder_name}_{run}.csv")
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Frequency (Hz)", "Trace |Z| (Ohm)"])
            # Write each frequency and impedance pair to the file
            for freq, z in zip(freq_Hz, impedance):
                writer.writerow([freq, z])
    else:
        print("Error: Insufficient data for Left and Right channels.")
    
    time.sleep(4)

# close the serial port
print("Closing serial port...")
serial_with_tympan.close()