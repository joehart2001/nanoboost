import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from nanoboost.scripts.feature_extraction.feature_extraction_utils import find_dwell_time_FWHM, resistive_conductive_area, decay_const, derivative_features, calculate_entropy, find_no_peaks_DNA, coeff_features

def find_features(event_time, event_data, mean_noise, coeffs, upper_threshold, lower_threshold = -np.inf, small_resistive = False, whole_run = False, DNA = False):
    """
    Extracts features from event data.

    Args:
        event_time (list of np.array): Time points for each event.
        event_data (list of np.array): Data points for each event.
        mean_noise (float): Mean noise level used for baseline correction.
        coeffs (list): List of wavelet coefficients for each event.
        upper_threshold (float): Upper threshold for peak detection.
        lower_threshold (float, optional): Lower threshold for trough detection. Defaults to -np.inf.
        small_resistive (bool, optional): Flag to adjust feature extraction for small resistive components. Defaults to False.
        whole_run (bool, optional): Flag to adjust feature extraction across the entire dataset. Defaults to False.
        DNA (bool, optional): Flag to adjust feature extraction specific to DNA sequences. Defaults to False.

    Returns:
        tuple: A tuple containing a DataFrame of extracted features and a list of feature arrays for each event.
    """
    # DNA has some bespoke features
    if DNA:            
        features_dic = {"deltaI_c": [], "dwell_time": [], "skew": [], "kurtosis": [], "conductive_area": [], "no_peaks":[], "peak_lower_delta_I":[], "peak_pos_rel":[], "entropy": [], "mean": [], "std": [], "var": [], "rms": [], "max_deriv": [], "min_deriv": [], "sum_absolute_changes": [], "sign_changes_sum": [], "decay_time_lhs": [], "decay_time_rhs": [], "approx_mean": [], "approx_sd": [], "approx_energy": [], "spectral_entropy": [], "band_power": []}
    # DWT applied on whole run has different features (no wavelet features)
    elif whole_run:
        features_dic = {"deltaI_c": [], "deltaI_r": [], "deltaI_c/deltaI_r": [], "dwell_time": [], "skew": [], "kurtosis": [], "total_area": [], "conductive_area": [], "resistive_area": [], "entropy": [], "mean": [], "std": [], "var": [], "rms": [], "max_deriv": [], "min_deriv": [], "sum_absolute_changes": [], "sign_changes_sum": [], "decay_time_lhs": [], "decay_time_rhs": []}
    # DWT applied to events
    else:
        features_dic = {"deltaI_c": [], "deltaI_r": [], "deltaI_c/deltaI_r": [], "dwell_time": [], "skew": [], "kurtosis": [], "total_area": [], "conductive_area": [], "resistive_area": [], "entropy": [], "mean": [], "std": [], "var": [], "rms": [], "max_deriv": [], "min_deriv": [], "sum_absolute_changes": [], "sign_changes_sum": [], "decay_time_lhs": [], "decay_time_rhs": [], "approx_mean": [], "approx_sd": [], "approx_energy": [], "spectral_entropy": [], "band_power": []}

    count = 0
    
    for event, time in zip(event_data, event_time):
        event_type = None
            
        # classify event as just conductive or conductive and resistive
        if small_resistive:
            event_type = "trough-peak"
            
        elif DNA:
            event_type = "peak-only"
            
        else:
            for i in range(len(event)):
                # Check for a trough-peak pattern
                if event[i] < -20 or event[i] < lower_threshold:
                    event_type = "trough-peak"
                    break
                # Check for a peak-only pattern
                elif event[i] > upper_threshold:
                    event_type = "peak-only"
                    break
            
        if not whole_run:
            # wavelet features
            features_list_approx = coeff_features(coeffs[count][0])
            features_dic["approx_mean"].append(features_list_approx[0])
            features_dic["approx_sd"].append(features_list_approx[1])
            features_dic["approx_energy"].append(features_list_approx[2])
            features_dic["spectral_entropy"].append(features_list_approx[3])
            features_dic["band_power"].append(features_list_approx[4])
            count += 1
        
        # dwell time
        dwell_time = find_dwell_time_FWHM(time, event, event_type)
        features_dic["dwell_time"].append(dwell_time)
        # decay
        decay_time_lhs, decay_time_rhs = decay_const(time, event, DNA = DNA)
        features_dic["decay_time_lhs"].append(decay_time_lhs)
        features_dic["decay_time_rhs"].append(decay_time_rhs)
        # charge
        conductive_area, resistive_area = resistive_conductive_area(time, event, event_type)
        features_dic["conductive_area"].append(conductive_area)
        if not DNA:
            features_dic["resistive_area"].append(resistive_area)
        #derivative features
        max_derivative, min_derivative, sum_abs_changes, sign_change_sum = derivative_features(time, event)
        features_dic["max_deriv"].append(max_derivative)
        features_dic["min_deriv"].append(min_derivative)
        features_dic["sum_absolute_changes"].append(sum_abs_changes)
        features_dic["sign_changes_sum"].append(sign_change_sum)
        # peak amplitudes + ratio
        deltaI_c = max(event) - mean_noise
        deltaI_r = min(event) - mean_noise
        deltaI_ratio = abs(deltaI_c / deltaI_r) if deltaI_r != 0 else 0
        features_dic["deltaI_c"].append(deltaI_c) 
        if not DNA:
            features_dic["deltaI_r"].append(deltaI_r)
            features_dic["deltaI_c/deltaI_r"].append(deltaI_ratio)
            
        if DNA:
            no_peaks, peak_pos_rel, peak_max_idx, peak_lower_idx = find_no_peaks_DNA(event, upper_threshold)
            
            features_dic["no_peaks"].append(no_peaks)
            features_dic["peak_pos_rel"].append(peak_pos_rel)
            features_dic["peak_lower_delta_I"].append(event[peak_lower_idx])
            
        entropy = calculate_entropy(event)
        features_dic["entropy"].append(entropy) 
        # statistical features
        features_dic["mean"].append(np.nanmean(event))
        features_dic["std"].append(np.nanstd(event))
        features_dic["var"].append(np.nanvar(event))
        features_dic["rms"].append(np.nanmean(np.sqrt(event**2)))
        features_dic["skew"].append(skew(event) if DNA else 0) 
        features_dic["kurtosis"].append(kurtosis(event)if DNA else 0)
        
        if not DNA:
            features_dic["total_area"].append(np.trapz(event, time)) # would be a duplicate feature for DNA
        
        df = pd.DataFrame(features_dic)
        
        features_list = []
        for i in range(len(df)): 
            features_list.append(list(df.iloc[i]))
                
    return df, features_list