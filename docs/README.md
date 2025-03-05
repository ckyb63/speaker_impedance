# Speaker Impedance

[![Version](https://img.shields.io/badge/Version-0.1.8-blue.svg)](CHANGELOG.md/#latest)
![Python Version](https://img.shields.io/badge/Python-3.12.9-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.8.1-blue.svg)
![Analog Discovery](https://img.shields.io/badge/Analog%20Discovery-2.0-green.svg)

## Overview

This repository contains the code and data for the speaker impedance project. There are 4 main components to the project:

1. **Analog Discovery** - This is the hardware device used to measure speaker impedance. The project includes two PyQt6-based GUIs:
   - `Auto_Impedance_PyQt6.py`: A modern dark-themed GUI for automated impedance data collection
   - `GUI_Predict.py`: A companion application for impedance measurement and length prediction

2. **Tympan** - Working with the Tympan library, this module is developed to allow the impedance measurement process to be done on a Tympan with the audio hat.

3. **Arduino** - Environmental monitoring system that reads temperature, humidity, and pressure data from the testing enclosure. The data is displayed in real-time in both GUIs and recorded in the measurement CSV files.

4. **Model** - AI model for predicting speaker tube length from impedance measurements, with support for different model architectures (DNet/CNet) and speaker differentiation.

## Key Features

### Measurement Interface

- Streamlined control panel with intuitive grouping
- Real-time impedance plotting with dark theme
- Environmental readings display (temperature, humidity, pressure)
- Automated data collection with progress tracking

### Prediction Interface

- CSV file loading and management
- Model type selection (DNet/CNet)
- Speaker differentiation option
- Confidence score display
- Length prediction results

### Data Collection

- Automatic organization of data by speaker type and length
- Environmental data logging with each measurement
- Consistent CSV format for training and prediction

## Repository Structure

For detailed information about each component, please refer to their respective documentation:

- [Prediction & Measurement App](../Prediction_Measurement_App/README.md) - GUI applications for data collection and prediction
- [Model Training](../Impedance-main/README.md) - AI model for length prediction
- [Standalone Prediction Script](../Impedance-main/predict_length_README.md) - Command-line prediction tool

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speaker_impedance.git
   cd speaker_impedance
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the WaveForms SDK from Digilent's website

4. Connect your hardware:
   - Connect the Analog Discovery 2 device via USB
   - (Optional) Connect an Arduino with BME280 sensor to monitor environmental conditions

5. Run either GUI application:
   ```bash
   python Analog\ Discovery/Auto_Impedance_PyQt6.py
   # or
   python Prediction_Measurement_App/GUI_Predict.py
   ```

## Data Collection Workflow

1. Configure measurement settings (speaker type, length, frequency range)
2. Connect the speaker to the Analog Discovery
3. Start measurement
4. Review collected data
5. (Optional) Run length prediction on the collected data

## Troubleshooting

- If the Analog Discovery is not detected, ensure WaveForms SDK is properly installed
- For Arduino communication issues, verify the correct COM port is selected
- For model loading errors, check the model file path and format

For more detailed instructions, see the [changelog](CHANGELOG.md) for recent updates and changes.

## Contributors

Max Chen
Keisuke Nakamura
