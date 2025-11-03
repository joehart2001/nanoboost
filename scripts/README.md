# Scripts

This directory contains Python scripts for data processing and model training.

## Contents

Scripts provided here can be used for:

- Data preprocessing (e.g., baseline correction, event isolation)
- Feature extraction from nanopore event data
- Discrete Wavelet Transform (DWT) implementations
- Machine learning model building with constrained minimization techniques
    - Custom metric definition and stochastic gradient descent optimization
    - Supervised learning models
    - Unsupervised learning models

## Integration

Scripts are designed to be modular and can be integrated into automated pipelines or used interactively from notebooks.


## Usage

#### Data Preprocessing

Baseline correction:
The baseline correction first involves importing the raw data into a manageable format then the simple moving average baseline correction can be performed. The Nanopore App includes a similar simple moving average baseline correction as one of its options.

Thresholding and Peak Finding:
Peak thresholds are found by a predetermined number of standard deviations of noise away from the mean. define_threshold calculates the mean and standard deviation of the noise before using them to calculate the threshold. This thresholding technique was inspired by The Nanopore App but adapted for a lower threshold for the detection of troughs.

Event Isolation:
Previous peak and next peak finding functions are used to pair up peaks and troughs. Event isolation works via peak width plus a buffer to make sure the entire event is isolated. If the event is too long, an even event isolation is applied on either side of the peak.
