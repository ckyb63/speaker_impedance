# Speaker Impedance

[![Version](https://img.shields.io/badge/Version-0.1.6-blue.svg)](CHANGELOG.md/#latest)
![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![Tkinter](https://img.shields.io/badge/Tkinter-8.6-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.8.1-blue.svg)
![Analog Discovery](https://img.shields.io/badge/Analog%20Discovery-2.0-green.svg)

## Overview

This repository contains the code and data for the speaker impedance project. There are 3 main parts to the project:

1. **Analog Discovery** - This is the device used to measure speaker impedance. The project includes two PyQt6-based GUIs:
   - `Auto_Impedance_PyQt6.py`: A modern dark-themed GUI for automated impedance data collection
   - `GUI_Predict.py`: A companion application for impedance measurement and length prediction

2. **Tympan** - Working with the Tympan library, this module is developed to allow the impedance measurement process to be done on a Tympan with the audio hat.

3. **Arduino** - Environmental monitoring system that reads temperature, humidity, and pressure data from the testing enclosure. The data is displayed in real-time in both GUIs and recorded in the measurement CSV files.

4. **Model** - AI model for predicting speaker tube length from impedance measurements, with support for different model architectures (DNet/CNet) and speaker differentiation.

## Recent Updates

- Modern dark theme UI with improved layout and usability
- Real-time environmental data display (temperature, humidity, pressure)
- Compact, efficient interface with horizontal layouts for advanced settings
- Improved speaker icon design for better visual representation
- Enhanced status indicators and progress tracking
- Automatic CSV file organization by speaker type and length
- Integrated prediction capabilities with model selection and confidence scoring

## Key Features

### Measurement Interface

- Streamlined control panel with intuitive grouping
- Real-time impedance plotting with dark theme
- Environmental readings display
- Automated data collection with progress tracking

### Prediction Interface

- CSV file loading and management
- Model type selection (DNet/CNet)
- Speaker differentiation option
- Confidence score display
- Length prediction results

For detailed information about each component, please refer to their respective documentation:

- [Prediction & Measurement App](Prediction_Measurement_App/README.md)
- [Model Training](project/README.md)
- [GPU Troubleshooting](project/GPU_TROUBLESHOOTING.md)

## Getting Started

1. Install required dependencies:

   ```bash
   pip install pyqt6 numpy pandas keras tensorflow scikit-learn pyserial matplotlib
   ```

2. Install the WaveForms SDK from Digilent
3. Connect your Analog Discovery 2 device
4. (Optional) Connect an Arduino with BME280 sensor to monitor environmental conditions
5. Run either GUI application:

   ```bash
   python Auto_Impedance_PyQt6.py
   # or
   python GUI_Predict.py
   ```

For more detailed instructions, see the [changelog](CHANGELOG.md) for recent updates and changes.
