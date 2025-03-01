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
- Model configuration options
- Confidence score visualization
- Length prediction display

## Requirements

- Python 3.10+
- PyQt6 6.8.1+
- numpy
- pandas
- keras
- tensorflow
- scikit-learn
- pyserial (for Arduino communication)
- matplotlib
- Analog Discovery 2 hardware and WaveForms SDK

## Installation

1. Install the required Python packages:

   ```bash
   pip install pyqt6 numpy pandas keras tensorflow scikit-learn pyserial matplotlib
   ```

2. Install the WaveForms SDK from [Digilent](https://digilent.com/reference/software/waveforms/waveforms-3/start)
3. Ensure the `dwf.dll` file is in your system path or application directory

## Usage

### Measurement Setup

1. Connect your Analog Discovery 2 device
2. (Optional) Connect an Arduino with BME280 sensor to COM8 (configurable)
3. Configure measurement settings:
   - Select earphone type and length
   - Set frequency range and steps
   - Adjust reference resistor value
4. Click "Start Measurement"

### Advanced Settings

- Start/Stop Frequency: Horizontal layout for frequency range
- Reference/Steps: Paired for efficient space usage
- Environmental readings: Compact grid layout
- Status indicators: Clear visual feedback

### Prediction Workflow

1. After measurement or via "Load CSV":
   - Select model type (DNet/CNet)
   - Configure speaker differentiation
   - Choose model file (optional)
2. View results:
   - Predicted length
   - Confidence score
   - File path confirmation

## Data Organization

- Measurements saved in `Collected_Data/[Type]_[Length]/`
- CSV format with environmental data
- Automatic file naming with run numbers

## Troubleshooting

### Hardware Connection

- Verify Analog Discovery 2 connection
- Check Arduino COM port settings
- Ensure WaveForms SDK installation

### Software Issues

- Confirm Python version compatibility
- Verify all dependencies installed
- Check model file format and location

### Environmental Sensors

- Monitor serial connection status
- Verify sensor readings in real-time
- Check CSV data recording

## Recent Updates

- Improved UI with dark theme
- Horizontal layout for advanced settings
- Enhanced speaker icon design
- Real-time environmental data integration
- Streamlined prediction workflow
- Improved status indicators

## Author

Max Chen

## Version

Current Version: 0.1.3 (February 2025)

See [CHANGELOG.md](../docs/CHANGELOG.md) for detailed update history.
