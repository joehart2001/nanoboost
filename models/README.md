# Models

This directory contains trained models and model configurations.

## Contents

- Trained NanoBoost models
- Model configuration files
- Model checkpoints and weights

## Usage

Models can be loaded and used for inference:

```python
import pickle

# Load a trained model
with open('models/nanoboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Model Information

Each model should include:
- Training date and version
- Performance metrics
- Hyperparameters used
- Training data description
