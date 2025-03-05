# Impedance Measurement and Length Prediction GUI

A modern, dark-themed application suite that combines impedance measurement using the Analog Discovery 2 with length prediction using a trained neural network model. The applications feature an intuitive, compact interface designed for efficient workflow.

## Features

### Modern UI Design

- Dark theme with consistent color scheme
- Compact, efficient layout with horizontal grouping
- Improved speaker icon and visual elements
- Real-time status indicators and progress tracking

### Measurement Panel

- Streamlined configuration with grouped settings
- Advanced settings arranged horizontally for better space utilization
- Real-time environmental data display (temperature, humidity, pressure)
- Live impedance plot with dark theme styling
- Automated data collection and organization

### Prediction Panel

- Immediate prediction after measurement
- CSV file management with clear status indicators
- Model configuration options (DNet/CNet)
- Confidence score visualization
- Length prediction display with error estimation

## Installation

1. Ensure you have Python 3.10+ installed on your system

2. Install the required Python packages:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Install the WaveForms SDK from [Digilent](https://digilent.com/reference/software/waveforms/waveforms-3/start)

4. Ensure the `dwf.dll` file is in your system path or application directory

5. Connect your hardware components:
   - Analog Discovery 2 via USB
   - Arduino (if using environmental monitoring)
   - Test speaker connected to Analog Discovery 2

## Usage

### Measurement Setup

1. Launch the application:

   ```bash
   python GUI_Predict.py
   ```

2. Configure measurement settings:
   - Select earphone type (A, B, C) and length
   - Set frequency range (default: 20Hz to 20kHz)
   - Adjust steps (default: 500)
   - Set reference resistor value (default: 1000Ω)

3. If using Arduino environmental monitoring:
   - Select the correct COM port
   - Click "Connect" to establish serial communication

4. Click "Start Measurement" to begin data collection

### Advanced Settings

- **Start/Stop Frequency**: Horizontal layout for frequency range definition
- **Reference/Steps**: Paired for efficient space usage and logical grouping
- **Environmental readings**: Compact grid layout with real-time updates
- **Status indicators**: Clear visual feedback on operation progress

### Prediction Workflow

1. After measurement or via "Load CSV":
   - Select model type (DNet/CNet)
   - Configure speaker differentiation if needed
   - Choose model file (optional, defaults to best model)

2. View results:
   - Predicted length in millimeters
   - Confidence score visualization
   - File path confirmation

## Data Organization

- Measurements saved in `Collected_Data/[Type]_[Length]/` directory structure
- CSV format with environmental data in the first line
- Columns: Frequency, Phase, Magnitude, Resistance, Reactance
- Automatic file naming with sequential run numbers
- Example: `A_5_Run1.csv`, `A_5_Run2.csv`, etc.

## Troubleshooting

### Hardware Connection

- **Analog Discovery 2 not detected**:
  - Verify USB connection
  - Confirm WaveForms SDK installation
  - Check if device appears in Device Manager

- **Arduino communication issues**:
  - Verify correct COM port selection
  - Check if Arduino appears in Device Manager
  - Ensure Arduino is running the correct firmware

### Software Issues

- **Model loading errors**:
  - Verify model file path
  - Check if model file exists and is not corrupted
  - Confirm model format is compatible (Keras .h5 or .keras format)

- **GUI rendering problems**:
  - Update graphics drivers
  - Install latest PyQt6 version
  - Check for Qt conflicts with other applications

### Environmental Sensors

- **Missing environmental data**:
  - Check Arduino serial connection
  - Verify BME280 sensor is properly connected
  - Test sensor independently with a simple sketch

## Recent Updates

- Improved UI with cohesive dark theme
- Horizontal layout for advanced settings (better space utilization)
- Enhanced speaker icon design
- Real-time environmental data integration
- Streamlined prediction workflow with confidence visualization
- Added error handling for hardware connection issues
- Improved CSV file management and organization

## Author

Max Chen

## Version

Current Version: 0.1.8 (March 2025)

See [CHANGELOG.md](../docs/CHANGELOG.md) for detailed update history.
