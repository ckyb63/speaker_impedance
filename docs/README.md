# Speaker Impedance

[![Latest Version](https://img.shields.io/badge/Latest-v0.9.1-blue.svg)](CHANGELOG.md/#latest)
[![Python Version](https://img.shields.io/badge/Python-3.12.9-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3129/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.8.1-blue.svg?logo=qt&logoColor=white)](https://pypi.org/project/PyQt6/)
[![Analog Discovery](https://img.shields.io/badge/Analog%20Discovery-2.0-green.svg?logo=digilent&logoColor=white)](https://digilent.com/reference/test-and-measurement/guides/waveforms-sdk-getting-started?srsltid=AfmBOorRtu33lsD6IVZflrbMJIFuTLurrbm7XozjjqH9yrPqBuhSF0tu)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-green.svg?logo=mathworks&logoColor=white)](https://www.mathworks.com/products/matlab.html)

## Overview

This repository contains the code and data for the speaker impedance research project. The following are the main components of the project:

1. **Analog Discovery** - The device of choice for this project is the Analog Discovery 2. This contains Automated impedance measurement application developed using PyQt6 with the WaveForms SDK.
   - `Auto_Impedance_New_GUI_PyQt6.py`: The latest newly developed GUI for automated impedance data collection.
   - Older versions of the GUI are located in the `Older` folder. The older GUI was developed using tkinter, with an identical application developed using PyQt6.

2. **Tympan** - Working with the Tympan library, this module is developed to allow the impedance measurement process to be done on a Tympan with the audio hat.

3. **Arduino Code** - Environmental monitoring system that reads temperature, and humiditydata from the testing enclosure. The data is displayed in real-time in the GUIs and recorded in the measurement CSV files.

4. **Impedance Main** - AI model for predicting speaker tube length from impedance measurements, with support for different model architectures (DNet/CNet) and speaker differentiation. See the [Impedance Main](../Impedance-main/README.md) for more details.

5. **Prediction Measurement App** - Standalone script for predicting speaker tube length from impedance measurements. The idea is that a measurement can be made and the length can be predicted immediately. See the [Prediction Measurement App](../Prediction_Measurement_App/README.md) for more details.

6. **MATLAB Code** - MATLAB code for plotting the collected impedance data for visualization.

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/ckyb63/speaker_impedance.git
   cd speaker_impedance
   ```

2. Install required dependencies, using the requirements.txt file located in the docs folder:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the WaveForms SDK from Digilent's website

4. Connect your hardware:
   - Connect the Analog Discovery 2 device via USB
   - (Optional) Connect an Arduino with BME280 sensor to monitor environmental conditions

5. Run the GUI application to make measurements:

   ```bash
   python Analog Discovery/Auto_Impedance_New_GUI_PyQt6.py
   ```

## Data Collection Workflow

1. Configure measurement settings (speaker type, length, frequency range) or keep it on default
2. Connect the speaker to the Analog Discovery
3. Start measurement
4. Review collected data

## Troubleshooting

- If the Analog Discovery 2 is not detected, ensure WaveForms SDK is properly installed and the device is connected via USB.
- For Arduino communication issues, verify the correct COM port is selected in the GUI.
- For model loading errors, check the model file path and format

## Updates

For more detailed updates, see the [changelog](CHANGELOG.md) for recent updates and changes.

## Contributors

Max Chen

Keisuke Nakamura
