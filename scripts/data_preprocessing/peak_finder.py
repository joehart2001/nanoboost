import numpy as np
from scipy.signal import find_peaks

def define_threshold(y_base, n_sd_upper, n_sd_lower = None):
    """
    Defines upper and optionally lower thresholds based on standard deviations from the mean.

    Args:
        y_base (np.array): Array of baseline-corrected signal values.
        n_sd_upper (float): Number of standard deviations above the mean to set the upper threshold.
        n_sd_lower (float, optional): Number of standard deviations below the mean to set the lower threshold. Defaults to None. Only used for resistive peaks.

    Returns:
        tuple: If `n_sd_lower` is provided, returns a tuple (upper threshold, lower threshold, mean, standard deviation).
               Otherwise, returns a tuple (upper threshold, mean, standard deviation).
    """
    
    sd_noise = np.std(y_base)
    mean_noise = np.mean(y_base)
    threshold = mean_noise + n_sd_upper * sd_noise
    
    # if a lower sd is provided for resistive peaks
    if n_sd_lower:
        threshold_lower = mean_noise - n_sd_lower * sd_noise
        return threshold, threshold_lower, mean_noise, sd_noise
    
    return threshold, mean_noise, sd_noise



def find_peaks_troughs(mean_noise, sd_noise, y_base, x_base, resistive = False, savefilename = None,  plot = False):
    """
    Identifies peaks and optionally troughs based on specified thresholds.

    Args:
        mean_noise (float): Mean value of the noise level in the signal.
        sd_noise (float): Standard deviation of the noise in the signal.
        y_base (np.array): Array of baseline-corrected signal values.
        x_base (np.array): Array of time or index values corresponding to y_base.
        resistive (bool, optional): If True, includes detection of troughs. Defaults to False.

    Returns:
        tuple: Returns thresholds and detected peaks and properties above (and optionally below) the thresholds.
               Structure is (sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below).
    """

    sd_threshold_lower = None
    
    # thresholds
    if resistive:
        sd_threshold, sd_threshold_lower, _, _ = define_threshold(y_base, 10, 10)
    else:
        sd_threshold, _, _ = define_threshold(y_base, 10)
    
    peaks_below = None
    properties_below = None
    
    # don't want to count a peak twice but don't want to miss if 2 peaks are nearby
    # distance equal to the average dwell time
    peaks_above, properties_above = find_peaks(y_base, height=sd_threshold, distance=200, width=30)
    
    # filter out peaks that are too narrow: don't spend adequate time above the threshold to fitler out noise spikes
    for i in peaks_above:
        for j in range(i, len(y_base)):
            if y_base[j] < sd_threshold:
                right_point = j
                break
        for j in range(i, 0, -1):
            if y_base[j] < sd_threshold:
                left_point = j
                break
        if right_point - left_point < 10:
            peaks_above = np.delete(peaks_above, np.where(peaks_above == i))
    
    
    
    if resistive:
        # Find peaks below the threshold (invert the signal)
        peaks_below, properties_below = find_peaks(-y_base, height=-sd_threshold_lower, distance=200, width=30)
    
    return sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below
