import numpy as np
import pywt

def wavelet_transform_func(signal, thresh, wavelet):
    """
    Applies a wavelet transform to a signal, thresholds the wavelet coefficients, and reconstructs the signal.

    Args:
        signal (np.array): The event current signal to transform.
        thresh (float, optional): Wavelet coefficient threshold
        wavelet (str, optional): Type of wavelet to use.

    Returns:
        tuple:
            - reconstructed_signal (np.array): The signal reconstructed from the thresholded wavelet coefficients.
            - coeffs (list of np.array): List of wavelet coefficients after thresholding.
    """
    thresh = thresh * np.nanmax(signal)
    coeffs = pywt.wavedec(signal, wavelet, mode="per")

    # Apply thresholding
    coeffs = [pywt.threshold(i, value=thresh, mode="soft") for i in coeffs] # was soft

    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode="per") # per should maintain length of signal but doesn't always
    
    if len(reconstructed_signal) > len(signal):
        reconstructed_signal = reconstructed_signal[:len(signal)]
    
    return reconstructed_signal, coeffs



def peak_tracer(event_data, DWT_sigs, threshold_upper, threshold_lower = None, thresh = 0.1, NRNS = False):
    """
    Takes 10 points around the peak and replaces the DWT signal with the raw signal if the DWT signal 
    is above the threshold in order to trace the peak

    Args:
    event_data (array-like): All events.
    DWT_sigs (array-like): DWT rec signal for all events.

    Returns:
    traced_data (array-like): Traced signal for all events.
    """
    
    traced_data = DWT_sigs.copy()
    range_offset = 100 if NRNS else 20 if thresh >= 0.2 else 10 # range around peak 
    
    for i in range(len(traced_data)):

        peak_above_idx = np.argmax(event_data[i])
        start_above_idx = max(0, peak_above_idx - range_offset)
        end_above_idx = min(len(event_data[i]), peak_above_idx + range_offset)
        idx_above = np.arange(start_above_idx, end_above_idx)

        for idx in idx_above:
            if event_data[i][idx] > threshold_upper:
                traced_data[i][idx] = event_data[i][idx]

        # trough trace
        if threshold_lower is not None :
            peak_below_idx = np.argmin(event_data[i])
            start_below_idx = max(0, peak_below_idx - range_offset)
            end_below_idx = min(len(event_data[i]), peak_below_idx + range_offset)
            idx_below = np.arange(start_below_idx, end_below_idx)
                        
            for idx in idx_below:
                if event_data[i][idx] < threshold_lower:
                    traced_data[i][idx] = event_data[i][idx]
                
    return traced_data