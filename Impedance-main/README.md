# Impedance Model Training

This directory contains the code for training, evaluating, and using neural network models to predict speaker tube length from impedance measurements.

## Directory Structure

### `config/`
Contains configuration files for training, dataset preparation, and model parameters. Modify these files to adjust settings for your specific use case.

Parameters include:
- Learning rate
- Batch size
- Number of epochs
- Input/output dimensions
- File paths

### `models/`
Contains model architecture definitions for different neural network designs:
- DNet: Default network architecture
- CNet: Alternative network with different layer configuration
- Custom models can be added here

### `utils/`
Contains helper functions for:
- Dataset creation and preprocessing
- Data normalization
- File handling
- Evaluation metrics

## Main Scripts

### `train_model.py`
The primary script for model training and evaluation.

```bash
python train_model.py --config config/train_config.json
```

Key features:
- Loads and preprocesses impedance data
- Trains model with specified architecture
- Evaluates performance on test set
- Saves trained model and metadata

### `transfer_learning.py`
Script for transfer learning, where a model is first trained on one dataset, then fine-tuned on another.

```bash
python transfer_learning.py --source_config config/source_config.json --target_config config/target_config.json
```

Benefits:
- Improves performance when target dataset is small
- Leverages knowledge from one speaker type to another
- Reduces training time for new models

### `cluster.py`
Performs K-means clustering on impedance data to identify patterns or groupings.

```bash
python cluster.py --data_path path/to/data --n_clusters 5
```

Applications:
- Dataset exploration
- Feature discovery
- Speaker type categorization

### `predict_length.py`
Standalone script for predicting tube length from CSV files.