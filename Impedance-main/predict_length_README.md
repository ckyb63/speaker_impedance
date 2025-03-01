# Predict Length Script

This is a simple, standalone script that uses a trained model to predict the length of a speaker tube based on impedance measurements in a CSV file.

## Usage

```bash
python predict_length.py --model path/to/model.keras --csv path/to/measurement.csv
```

### Arguments:

- `--model`: Path to the trained model file (default: 'best_model.keras')
- `--csv`: Path to the CSV file to predict (required)
- `--model_type`: Type of model, either 'DNet' or 'CNet' (default: 'DNet')
- `--speaker_diff`: Enable speaker differentiation in labels (flag, default: disabled)

## Examples

### Basic usage:

```bash
python predict_length.py --model best_model.keras --csv Collected_Data_Sep16/A/A_5/A_5_Run1.csv
```

### Using CNet model:

```bash
python predict_length.py --model trained_model.keras --csv Collected_Data_Sep16/B/B_11/B_11_Run2.csv --model_type CNet
```

### With speaker differentiation:

```bash
python predict_length.py --model best_model.keras --csv Collected_Data_Sep16/C/C_17/C_17_Run3.csv --speaker_diff
```

## Output

The script will output:
- The CSV file path
- The predicted length
- The confidence score (probability) of the prediction

## Notes

- The script automatically handles the preprocessing of the CSV file, including normalization.
- It assumes the CSV file has the same format as the training data, with columns for frequency, phase, magnitude, resistance, and reactance.
- The default configuration matches the settings used during training, but you can modify the script if needed.
- If you're using a model trained with speaker differentiation, make sure to use the `--speaker_diff` flag. 