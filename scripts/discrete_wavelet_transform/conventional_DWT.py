import numpy as np
import pywt
from nanoboost.scripts.data_preprocessing.baseline_correction import importABF_movingavg
from nanoboost.scripts.discrete_wavelet_transform.wavelet_transform_setup import wavelet_transform_func
from nanoboost.scripts.data_preprocessing.peak_finder import define_threshold, find_peaks_troughs
from nanoboost.scripts.data_preprocessing.event_isolation import event_isolation


def load_to_event_DWT_wholeevent_isolation(path, thresh, wavelet, resistive = False, plot = False):
    """
    Loads data from a specified path, applies a Discrete Wavelet Transform (DWT) for noise reduction to the whole trace, and then event isolation to the wavelet-transformed data.

    Args:
        path (str): Path to the data file.
        thresh (float): Threshold factor for wavelet coefficient thresholding.
        wavelet (str): Type of wavelet to use in the transformation.
        resistive (bool, optional): If True, considers resistive peaks in the event isolation process. Defaults to False.
        plot (bool, optional): If True, plots the detected peaks and troughs during processing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - event_time (list of np.array): Time indices for each isolated event.
            - event_data (list of np.array): Data points for each isolated event.
            - DWT_rec (np.array): The reconstructed signal from the DWT.
            - sd_threshold (float): Standard deviation threshold used for peak detection.
            - sd_threshold_lower (float): Lower standard deviation threshold used for peak detection, if applicable.
            - mean_noise (float): Calculated mean noise level in the data.
            - peaks_above (np.array): Indices of detected peaks above the threshold.
            - peaks_below (np.array): Indices of detected troughs below the threshold, if resistive is True.
            - all_coeffs (list): List of wavelet coefficients after thresholding.
    """
    
    # import file
    x, y, sma, y_corrected, y_base, x_base = importABF_movingavg(path, resistive) 

    threshold, mean_noise, sd_noise = define_threshold(y_base, 12)
    
    # DWT
    DWT_rec, all_coeffs = wavelet_transform_func(y_base, thresh=thresh, wavelet=wavelet)
    DWT_rec, all_coeffs = DWT_rec[:len(y_base)], list(all_coeffs) # didnt convert to list as find_peaks_troughs takes an array
    # finding peaks
    sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below = find_peaks_troughs(mean_noise, sd_noise, DWT_rec, x_base, resistive, plot = plot)
    # event isolation
    if resistive:
        event_time, event_data, smo = event_isolation(x_base, DWT_rec, peaks_above, properties_above, peaks_below, properties_below)
    else:
        event_time, event_data, smo = event_isolation(x_base, DWT_rec, peaks_above, properties_above)        
    
    return event_time, event_data, DWT_rec, sd_threshold, sd_threshold_lower, mean_noise, peaks_above, peaks_below, all_coeffs