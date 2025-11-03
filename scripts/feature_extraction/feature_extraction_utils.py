import numpy as np
from scipy import signal
from collections import Counter
from scipy.stats import entropy
from scipy.signal import find_peaks

def find_dwell_time_FWHM(time, event, event_type):
    
    """ Finds the dwell time of an event using the full width at half maximum (FWHM) method
    
    Args:
        time (numpy.ndarray): The time array corresponding to one event.
        event (numpy.ndarray): The current array for the event.
        event_type (str): Type of event to analyze. Accepts "trough-peak" or "peak-only".

    Returns:
        float: The dwell time of the event, calculated as the difference in time between the start and end points defined by the FWHM method.

    """

    start, end = 0, 0 # Initialize
    
    if event_type == "trough-peak" and event[0] >= np.min(event)/2 and event[-1] <= np.max(event)/2:
        # condition prevents error in the case where the start and end of the event are not below the half max value
        half_max_peak = np.max(event)/2
        intercept_peak = np.where(np.diff(np.sign(event - half_max_peak)))[0] # find where the peak signal crosses the half max value
        half_max_trough = np.min(event)/2
        intercept_trough = np.where(np.diff(np.sign(event - half_max_trough)))[0] # find where the trough signal crosses the half max value

        start = intercept_trough[intercept_trough < np.argmin(event)]
        start = start[-1] if start.size != 0 else 0 # 
        end = intercept_peak[intercept_peak > np.argmax(event)]
        end = end[0] if end.size != 0 else len(event) - 1
        
    elif event_type == "peak-only" and event[0] <= np.max(event)/2 and event[-1] <= np.max(event)/2:
        half_max_peak = np.max(event)/2
        intercepts = np.where(np.diff(np.sign(event - half_max_peak)))[0] # find where the signal crosses the half max value
        
        start = intercepts[intercepts < np.argmax(event)]
        start = start[-1] if start.size != 0 else 0
        end = intercepts[intercepts > np.argmax(event)]
        end = end[0] if end.size != 0 else len(event) - 1
    
    else:
        start = 0
        end = len(event) - 1
        
    dwell_time = time[end] - time[start]
        
    return dwell_time



def resistive_conductive_area(time, event, event_type):
    """
    Calculates the resistive and conductive areas under the curve for specified event types by identifying zero crossings.

    Args:
        time (np.array): Array of time points corresponding to the event data points.
        event (np.array): Array of event data points, typically voltage or current over time.
        event_type (str): Type of event to analyze, can be 'trough-peak' for events with a clear trough and peak, or 'peak-only' for events with only a peak.

    Returns:
        tuple: A tuple (conductive_area, resistive_area) where:
            - conductive_area (float): The integrated area of the event from a zero crossing to the end for 'trough-peak', or the total area for 'peak-only'.
            - resistive_area (float): The integrated area from the start of the event to a zero crossing for 'trough-peak', or from the start to the last zero crossing before the peak for 'peak-only'.
    """
    crossing_index = 0
    resistive_area = 0
    conductive_area = 0
    
    # find zero crossings
    zero_crossings = np.where(np.diff(np.sign(event)))[0]
    
    # resistive and conductive area
    if event_type == "trough-peak":
        peak = np.argmax(event)
        trough = np.argmin(event)
        
        for crossing in zero_crossings:
            if trough < crossing < peak:
                crossing_index = crossing
                break
    
        conductive_area = np.trapz(event[crossing_index:], time[crossing_index:])
        resistive_area = np.trapz(event[:crossing_index], time[:crossing_index])
        
    # just conductive area (overall area)
    if event_type == "peak-only":
        peak = np.argmax(event)
        conductive_area = np.trapz(event, time)
        
        # sometimes the resistive part doesn't trigger the treshold but is still present (but small)
        for i in range(len(zero_crossings), 0, -1):
            if zero_crossings[i-1] < peak:
                resistive_area = np.trapz(event[:zero_crossings[i-1]], time[:zero_crossings[i-1]])
                break

    return conductive_area, resistive_area



def decay_const(time, event, DNA = False):
    """
    Calculates the decay constant for the peak and trough or just peak if no trough present.

    Args:
        time (np.array): Array of time points corresponding to the event data points.
        event (np.array): Array of event data points.
        DNA (bool, optional): If True, adjusts the decay calculation to be specific for DNA-related analyses. Defaults to False.

    Returns:
        tuple: Returns a tuple (decay_const_lhs, decay_const_rhs) representing the decay constant to the left and right of the peak, respectively.
    """
    peak = np.argmax(event)
    current_data_rhs = event[peak:]
    time_data_rhs = time[peak:]
    
    # Find the decay constant on the right-hand side
    rhs_condition = current_data_rhs < current_data_rhs[0] / np.e
    decay_const_rhs = time_data_rhs[np.argmax(rhs_condition)] - time_data_rhs[0] if rhs_condition.any() else 0

    decay_const_lhs = 0
    lhs_condition = None

    # data containing troughs
    if peak > 120 and np.min(event) < -20 and event[0] >= np.min(event) / np.e:         
        trough = np.argmin(event[:peak])
        current_data_lhs = event[:trough]
        time_data_lhs = time[:trough]
        
        if len(current_data_lhs) > 0:
            lhs_condition = current_data_lhs > current_data_lhs[-1] / np.e
        else:
            lhs_condition = np.array([False])

        # Find the decay constant on the left-hand side if applicable
        if lhs_condition.any():
            reverse_index = len(lhs_condition) - np.argmax(lhs_condition[::-1]) - 1
            decay_const_lhs = time_data_lhs[-1] - time_data_lhs[reverse_index]
            
        else: 
            decay_const_lhs = 0
    elif DNA:
        peak = np.argmax(event)
        current_data_lhs = event[:peak]
        time_data_lhs = time[:peak]
        
        # Find the decay constant on the right-hand side
        lhs_condition = current_data_lhs < current_data_lhs[0] / np.e
        decay_const_lhs = time_data_lhs[np.argmax(lhs_condition)] - time_data_lhs[0] if lhs_condition.any() else 0
        
    
    else:
        decay_const_lhs = 0  # To avoid errors if lhs_condition is not set
        
    return decay_const_lhs, decay_const_rhs



def calculate_entropy(event):
    """
    Calculates the entropy of an event.

    Args:
        event (np.array): The event data from which to calculate entropy. Data points are rounded to the nearest integer.

    Returns:
        float: The calculated entropy of the event.
    """
    counter_values = Counter(np.round(event, 0)).most_common()
    probabilities = [elem[1]/len(event) for elem in counter_values]
    entrop =entropy(probabilities)
    return entrop



def derivative_features(time, event):
    """
    Computes derivative-based features of an event, including maximum and minimum derivatives, 
    total sum of absolute changes, and count of sign changes.

    Args:
        time (np.array): Array of time points corresponding to the event data points.
        event (np.array): Array of event data points, typically voltage or current over time.

    Returns:
        tuple: A tuple containing the following derivative features:
            - max_deriv (float): Maximum derivative value of the event.
            - min_deriv (float): Minimum derivative value of the event.
            - sum_absolute_changes (float): Sum of the absolute changes in the derivative.
            - sign_changes (int): Count of the times the sign of the derivative changes.
    """
    time_diff = np.diff(time)
    if np.any(time_diff == 0):
        # replace with 2e-6, sampling rate used in experiment
        time_diff[time_diff == 0] = 2e-6        
    current_deriv = np.diff(event) / time_diff
    
    max_deriv = np.max(current_deriv)
    min_deriv = np.min(current_deriv)
    sum_absolute_changes = np.sum(np.abs(current_deriv))
    sign_changes = np.sum(np.diff(np.sign(current_deriv)) != 0)
    
    return max_deriv, min_deriv, sum_absolute_changes, sign_changes



# wavelet coefficient features
def coeff_features(approx_coeffs): 
    """
    Computes statistical and spectral features from wavelet approximation coefficients.

    Args:
        approx_coeffs (np.array): Wavelet approximation coefficients of an event signal.

    Returns:
        list: A list of computed features including mean, standard deviation, energy,
              spectral entropy, and total band power of the coefficients.
    """
    features_approx = []
    
    approx_mean = np.mean(approx_coeffs)
    approx_sd = np.std(approx_coeffs)
    approx_energy = np.sum(np.square(approx_coeffs))
    
    # Welch's periodogram method to calculate PSD
    freqs_approx, psd_approx = signal.welch(approx_coeffs, fs=1/(2e-6), nperseg=len(approx_coeffs))
    epsilon_appprox = 1e-10 # to prevent log2(0) error
    spectral_entropy = -np.sum(psd_approx * np.log2(psd_approx + epsilon_appprox))
    band_power = np.sum(psd_approx)

    features_approx.extend([approx_mean, approx_sd, approx_energy, spectral_entropy, band_power])

    return features_approx 



def coeff_features_dict(features_approx):
    event_features___ = {}
    event_features___['approx'] = features_approx
        
    return event_features___



def find_no_peaks_DNA(event, sd_threshold):
    """
    Identifies the number of peaks in the DNA event after the primary peak based on specified thresholds.

    Args:
        event (np.array): Event data points, typically representing a signal.
        sd_threshold (float): Standard deviation threshold used to define secondary peak detection criteria.

    Returns:
        tuple: A tuple containing:
            - no_peaks (int): Number of detected peaks including the primary peak.
            - peak_pos_rel (float): Relative position of the primary peak in the event.
            - peak_max (int): Index of the primary peak in the event.
            - peak_lower_idx (int): Index of the secondary peak if found; otherwise, index of the primary peak.
    """
    # overall event maximum
    peak_max = np.argmax(event) 
    # find any peaks below max, but not too close otherwise will detect the edges of peak max
    peaks_lower, _ = find_peaks(event[peak_max+20:], height=(sd_threshold, event[peak_max] - 20), distance=50, prominence=0.5)
        
    if peaks_lower.size > 0:
        no_peaks = 2
        peak_lower_idx = peaks_lower[0] + peak_max + 20
    else:
        no_peaks = 1
        peak_lower_idx = peak_max
        
    peak_pos_rel = peak_max / len(event)
    
    return no_peaks, peak_pos_rel, peak_max, peak_lower_idx
