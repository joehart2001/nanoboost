import numpy as np
from nanoboost.scripts.data_preprocessing.baseline_correction import importABF_movingavg
from nanoboost.scripts.data_preprocessing.peak_finder import define_threshold, find_peaks_troughs
from nanoboost.scripts.data_preprocessing.event_isolation import event_isolation, event_isolation_NRNS, pad_event
from nanoboost.scripts.discrete_wavelet_transform.wavelet_transform_setup import wavelet_transform_func, peak_tracer
from nanoboost.scripts.feature_extraction.event_feature_extraction import find_features

def load_to_event_data_nofeatures(path, resistive = False, plot = False, NRNS = False, NP = True):
    """
    Loads and processes data from a raw data to isolated events.

    Args:
        path (str): Path to the data file.
        resistive (bool, optional): If True, considers resistive peaks during event isolation. Defaults to False.
        plot (bool, optional): If True, plots are generated during peak detection. Defaults to False.
        NRNS (bool, optional): If True, uses a non-resistive/non-specific event isolation function. Defaults to False.
        NP (bool, optional): If True and NRNS is True, adjusts the non-peak isolation strategy. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - event_time (list of np.array): Time indices for each isolated event.
            - event_data (list of np.array): Data points for each isolated event.
            - smo (list or None): Smoothed data or None, depending on the isolation strategy.
            - sd_threshold (float): Standard deviation threshold used for peak detection.
            - sd_threshold_lower (float): Lower standard deviation threshold, if applicable.
            - mean_noise (float): Calculated mean noise level in the data.
    """
    x, y, sma, y_corrected, y_base, x_base = importABF_movingavg(path, resistive) 
    
    threshold, mean_noise, sd_noise = define_threshold(y_base, 10)
    
    sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below = find_peaks_troughs(mean_noise, sd_noise, y_base, x_base, resistive, plot = plot)
    
    if NRNS: # option for different event isolation function
        event_time, event_data = event_isolation_NRNS(x_base, y_base, peaks_above, NP = NP)
        smo = None
    if resistive:
        event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above, peaks_below, properties_below)
    else:
        event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above)        
    
    
    return event_time, event_data, smo, sd_threshold, sd_threshold_lower, mean_noise





def DWT_and_features_thresh_trace(event_time, event_data, mean_noise, sd_threshold, sd_threshold_lower, NP_label, wavelet, threshold, NRNS = False, DNA = False):
    """
    Performs the DWT with peak tracing and extracts features

    Args:
        event_time, event_data (list of arrays): event data for 1 run, containing all events 
        mean_noise, sd_threshold, sd_threshold_lower, NP_size, wavelet, threshold

    Returns:
        DWT_rec, event_time_padded, DWT_rec_padded, features_df, features_list, labels, all_coeffs
    """
    
    # DWT
    DWT_rec, all_coeffs = zip(*[wavelet_transform_func(signal, wavelet=wavelet, thresh = threshold) for signal in event_data])
    DWT_rec, all_coeffs = list(DWT_rec), list(all_coeffs)

    # trace peak and optionally trough
    DWT_rec = peak_tracer(event_data, DWT_rec, sd_threshold, sd_threshold_lower, thresh = threshold, NRNS = NRNS)
    
    # pad data
    if NRNS:
        event_time_padded, DWT_rec_padded = pad_event(event_time, DWT_rec, mean_noise, custom_length=2000)
    elif DNA:
        event_time_padded, DWT_rec_padded = pad_event(event_time, DWT_rec, mean_noise, custom_length=501)
    else:
        event_time_padded, DWT_rec_padded = pad_event(event_time, DWT_rec, mean_noise, custom_length=1000)
    features_df, features_list = find_features(event_time_padded, DWT_rec_padded, mean_noise, all_coeffs, sd_threshold, sd_threshold_lower)
    
    # assign label 0 for 5nm or NS, 1 for 10nm or NR
    if DNA:
        labels = None
    else:
        labels = np.ones(len(features_list)) if NP_label in [10, "NR"] else np.zeros(len(features_list))
    
    
    return DWT_rec, event_time_padded, DWT_rec_padded, features_df, features_list, labels, all_coeffs