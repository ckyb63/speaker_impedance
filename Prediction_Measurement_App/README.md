# Impedance Measurement and Length Prediction GUI

This application combines impedance measurement using the Analog Discovery 2 with length prediction using a trained neural network model. It allows users to perform impedance measurements on earphone tubes and predict their length based on the measured impedance characteristics.

## Features

### Measurement Capabilities
- Configure measurement parameters (frequency range, steps, reference resistor)
- Select earphone type and length for data organization
- Display real-time environmental data (temperature, humidity, pressure) from Arduino
- Visualize impedance measurements in real-time with a log-log plot
- Save measurement data to CSV files

### Prediction Capabilities
- Load existing CSV files or use freshly measured data
- Select model type (DNet or CNet)
- Enable/disable speaker differentiation
- Display prediction results with confidence scores

## Requirements

- Python 3.6+
- PyQt6
- matplotlib
- numpy
- pandas
- keras
- tensorflow
- scikit-learn
- pyserial (for Arduino communication)
- Analog Discovery 2 hardware and WaveForms SDK

## Installation

1. Install the required Python packages:
   ```
   pip install pyqt6 matplotlib numpy pandas keras tensorflow scikit-learn pyserial
   ```

2. Install the WaveForms SDK from Digilent: https://digilent.com/reference/software/waveforms/waveforms-3/start

3. Make sure the `dwf.dll` file is in your system path or in the same directory as the application.

## Usage

### For Measurements:

1. Connect your Analog Discovery 2 device to your computer
2. (Optional) Connect an Arduino with environmental sensors to COM8 (or modify the code for your COM port)
3. Select the earphone type and length from the dropdown menus
4. Configure the frequency range, steps, and reference resistor value
5. Click "Start Measurement" to begin the impedance measurement
6. The application will display the measurement progress and plot the impedance curve

### For Predictions:

1. After a measurement is complete, the application will automatically switch to the prediction tab
2. Alternatively, you can load an existing CSV file by clicking "Load CSV File"
3. Select the model type (DNet or CNet) based on your trained model
4. Check "Enable Speaker Differentiation" if your model was trained with speaker differentiation
5. Click "Predict Length" to get the prediction result
6. The application will display the predicted length and confidence score

## Model Files

The application looks for model files in the following locations:
1. First, it checks the local `Model` directory for `.keras` or `.h5` files
2. If no model is found, it falls back to `../Impedance-main/best_model.keras`

You can also manually select a model file using the "Browse" button in the Prediction tab.

## Data Storage

Measurement data is stored in the `Collected_Data` directory, organized by earphone type and length.

## Troubleshooting

- If the application fails to connect to the Analog Discovery 2, make sure the device is properly connected and the WaveForms SDK is installed.
- If the application fails to load the model, check that the model file exists and is in the correct format.
- If the Arduino connection fails, check the COM port setting in the code.

## Author

Max Chen