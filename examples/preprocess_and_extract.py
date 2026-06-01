"""
Example: preprocess an ABF recording and extract wavelet features.

Demonstrates the two-stage pipeline:
  1. load_to_event_data_nofeatures  — baseline correction, peak detection, event isolation
  2. DWT_and_features_thresh_trace  — per-event DWT (bior3.3, threshold=0.2) + feature extraction

Usage:
    python examples/preprocess_and_extract.py
    python examples/preprocess_and_extract.py path/to/recording.abf 10
"""

import sys
from pathlib import Path

from nanoboost.scripts.discrete_wavelet_transform.novel_DWT import (
    DWT_and_features_thresh_trace,
    load_to_event_data_nofeatures,
)

# --- configuration -----------------------------------------------------------
ABF_PATH = Path(__file__).resolve().parent.parent / "tests" / "5nm_test.abf"
NP_LABEL = 5        # 5 or 10 (nm nanospheres), or "NR" / "NS" (nanorods / nanospheres)
WAVELET   = "bior3.3"  # paper-optimal mother wavelet
THRESHOLD = 0.2        # paper-optimal DWT coefficient threshold

if len(sys.argv) >= 2:
    ABF_PATH = Path(sys.argv[1])
if len(sys.argv) >= 3:
    arg = sys.argv[2]
    NP_LABEL = int(arg) if arg.isdigit() else arg

# --- stage 1: preprocess -----------------------------------------------------
print(f"Loading:  {ABF_PATH.name}")
event_time, event_data, _, sd_threshold, sd_threshold_lower, mean_noise = \
    load_to_event_data_nofeatures(str(ABF_PATH), resistive=False)

print(f"Isolated: {len(event_data)} events")
print(f"Threshold:  {sd_threshold:.2f} pA  (mean noise: {mean_noise:.2f} pA)")

if not event_data:
    print("No events found — try a different recording.")
    sys.exit(0)

# --- stage 2: DWT + feature extraction ---------------------------------------
print(f"\nApplying DWT ({WAVELET}, threshold={THRESHOLD}) ...")
_, _, _, features_df, _, labels, _ = DWT_and_features_thresh_trace(
    event_time, event_data, mean_noise,
    sd_threshold, sd_threshold_lower,
    NP_LABEL, WAVELET, THRESHOLD,
)

print(f"Extracted: {len(features_df)} events × {len(features_df.columns)} features")
print(f"Labels:    {labels.tolist()}")
print()
print(features_df.to_string())
