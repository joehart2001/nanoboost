# Python file containing all libraries and (most) functions
# to be imported to any jupyter notebook


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
from scipy.signal import find_peaks, peak_widths
import pyabf
import pyabf.filter
from multiprocessing import Pool
import pywt

from sklearn.inspection import permutation_importance
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis
import seaborn as sns
from scipy.stats import randint
from scipy.signal import welch

from sklearn.metrics import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator
import sklearn
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from scipy.stats import entropy
from collections import Counter

from sklearn.base import is_classifier, clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from tslearn.clustering import TimeSeriesKMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import pickle
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 12})



def compute_moving_average(y, window_size):
    """
    Computes the simple moving average of a given numerical sequence.

    Args:
        y (array): Raw current values.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Array of the simple moving averages.
    """
    weights = np.ones(window_size) / window_size  # normalise array so sum of weights = 1, window size for moving average
    sma = np.convolve(y, weights, 'valid') # computes convolution of two sequences (combines two signals to compute a third) to compute the moving average

    return sma



def importABF_movingavg(path, resistive):
    """
    Processes ABF file data with a moving average and baseline correction.

    Args:
        path (str): Path to the ABF file to be processed.
        resistive (bool): Indicates whether to preserve negative values in the output.

    Returns:
        tuple: A tuple containing the time values (x), raw y-values (y), moving average (sma),
               corrected y-values (y_corrected), baseline adjusted y-values (y_base),
               and base time values (x_base).
    """
    abf = pyabf.ABF(path)
    # select which sweep (or trial) from the ABF file, as there may be many sweeps

    abf.setSweep(0)
        
    x = abf.sweepX
    y = -abf.sweepY
    
    x = x[x<60] # 60 second recording
    y = y[:len(x)]
    
    # moving average
    window_size = 10000
    sma = compute_moving_average(y, window_size)
    
    
    #adjust x and y to match the size of sma
    x = x[:len(sma)]
    y = y[:len(sma)]
    
    y_corrected = y - sma #correct for baseline drift
    
    
    if resistive:
        # don't remove signal below baseline
        y_base = y_corrected        
                
    else:
        # make all negative values = 0
        y_base = np.maximum(y_corrected, 0) 
        
    x_base = x
    
    return x, y, sma, y_corrected, y_base, x_base


# dont need
def plot_baseline_correct(x, y, sma, y_corrected, y_base, x_base, filename = None, ylim = None):
    plt.figure(figsize=(10, 3))

    # First plot
    plt.subplot(1, 2, 1)  # (rows, columns, panel number)
    plt.plot(x, y)
    plt.plot(x, sma, label='Moving Average', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)')
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(860, 1120)
    #plt.title('Raw Data')
    plt.legend(loc = "upper right")

    # # Second plot
    # plt.subplot(1, 3, 2)
    # plt.plot(x, y_corrected)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Current (pA)')
    # plt.title('Baseline drift corrected')

    # Third plot
    plt.subplot(1, 2, 2)  
    plt.plot(x_base, y_base)
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)')
    #plt.title('Baseline Corrected')

    plt.tight_layout()
    if filename:
        plt.savefig(f'MSci_python_images/{filename}.png', dpi=300)
    plt.show()
    
def plot_baseline_correct_resistive(x, y, sma, y_base, x_base, filename = None):
    plt.figure(figsize=(10, 3)) 

    # First plot
    plt.subplot(1, 2, 1)  # (rows, columns, panel number)
    plt.plot(x, y)
    plt.plot(x, sma, label='Moving Average', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)')
    plt.ylim(160, 450)
    #plt.title('Raw Data')
    plt.legend(loc = "upper right")
    plt.tight_layout()

    # Second plot
    plt.subplot(1, 2, 2)
    plt.plot(x_base, y_base)
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)', labelpad = -5)
    plt.tight_layout()
    #plt.title('Baseline Corrected')

    # Adjust the layout so the plots are not overlapping
    plt.tight_layout()
    if filename:
        plt.savefig(f'/Users/joehart/Desktop/chemistry/Year 4/MSci project/Python_nanopores/MSci_python_images/{filename}.png', dpi=300)
    plt.show()
    
    
    
    
# def define_threshold(y_base, n_sd_upper, n_sd_lower = None):
#     sd_noise = np.std(y_base)
#     mean_noise = np.mean(y_base)
#     threshold = mean_noise + n_sd_upper * sd_noise
    
#     if n_sd_lower:
#         threshold_lower = mean_noise - n_sd_lower * sd_noise
#         return threshold, threshold_lower, mean_noise, sd_noise
    
#     return threshold, mean_noise, sd_noise

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



def find_peaks_troughs(mean_noise, sd_noise, y_base, x_base, resistive = False, savefilename = None,  plot = False):
    sd_threshold_lower = None
    
    if resistive:
        sd_threshold, sd_threshold_lower, _, _ = define_threshold(y_base, 12, 8)
    else:
        sd_threshold, _, _ = define_threshold(y_base, 12)
    
    peaks_below = None
    properties_below = None
    
    # don't want to count a peak twice but don't want to miss if 2 peaks are nearby
    # want to make distance equal to the average dwell time
    peaks_above, properties_above = find_peaks(y_base, height=sd_threshold, distance=200, width=30)
    
    # filter out peaks that are too narrow
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
        #sd_threshold_lower = mean_noise - 12*sd_noise
        # Find peaks below the threshold (invert the signal)
        peaks_below, properties_below = find_peaks(-y_base, height=-sd_threshold_lower, distance=200, width=30)

        if plot:
            plt.figure(figsize=(4, 2))
            plt.plot(x_base, y_base)
            plt.axhline(y=sd_threshold, color='r', linestyle='--', label='Threshold')
            plt.axhline(y=sd_threshold_lower, color='r', linestyle='--')
            plt.plot(x_base[peaks_above], y_base[peaks_above], "x", label='Peaks Above')
            plt.plot(x_base[peaks_below], y_base[peaks_below], "x", label='Peaks Below')
            plt.ylim(-150, 150)
            plt.xlabel('Time (s)')
            plt.ylabel('Current (pA)')
            plt.legend(loc = "upper right")
            plt.tight_layout()
            plt.show()

    else:
        if plot:
            plt.figure(figsize=(4, 2))
            plt.plot(x_base, y_base, label='Current vs. Time')
            plt.axhline(y=sd_threshold, color='r', linestyle='--', label='Threshold')
            plt.plot(x_base[peaks_above], y_base[peaks_above], "x", label='Peaks Above')
            plt.ylim(-10, 170)
            plt.xlabel('Time (s)')
            plt.ylabel('Current (pA)')
            plt.legend(loc = "upper right")
            plt.tight_layout()
            plt.show()

    
    if savefilename:
        plt.savefig(f'MSci_python_images/{savefilename}.png', dpi=300, bbox_inches='tight')
    
    
    return sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below






def find_previous_peak(trough, peaks):
    """
    Finds the last peak that occurs before a specified trough.

    Args:
        trough (int): The index of the trough in the data.
        peaks (np.array): An array of indices representing the locations of peaks.

    Returns:
        int or None: The index of the nearest peak before the trough, or None if there are no previous peaks.
    """
    previous_peaks = peaks[peaks < trough]
    if len(previous_peaks) == 0:
        return None
    return previous_peaks[-1]

def find_next_peak(trough, peaks):
    """
    Identifies the first peak that occurs after a specified trough.

    Args:
        trough (int): The index of the trough in the data.
        peaks (np.array): An array of indices representing the locations of peaks.

    Returns:
        int or None: The index of the nearest peak after the trough, or None if there are no subsequent peaks.
    """
    next_peaks = peaks[peaks > trough]
    if len(next_peaks) == 0:
        return None
    return next_peaks[0]

def get_event_bounds(center_idx, widths, data_length, buffer=200):
    """
    Calculates the start and end indices for an event centered at a specified index, plus a buffer.

    Args:
        center_idx (int): The index of the event center.
        widths (np.array): Array containing the widths of events at their respective indices.
        data_length (int): The total number of data points.
        buffer (int, optional): Additional space around the event width. Defaults to 200.

    Returns:
        tuple: A tuple containing the start and end indices of the event.
    """
    width = widths[center_idx]
    start = max(0, center_idx - int(width / 2) - buffer)
    end = min(data_length, center_idx + int(width / 2) + buffer)
    return start, end




def event_isolation(x_base, y_base, peaks_above, properties_above, peaks_below = None, properties_below = None):
    """
    Isolates events from signal data based on detected peaks and optionally troughs, producing events of a specified max length.

    Args:
        x_base (np.array): Time indices or equivalent sequential data points.
        y_base (np.array): Baseline-corrected signal values.
        peaks_above (np.array): Peak indices.
        properties_above (dict): Properties of the peaks, such as widths, must include 'widths'.
        peaks_below (np.array, optional): Trough indices if any, defaults to None.
        properties_below (dict, optional): Properties of the troughs, such as left_ips and right_ips if available, defaults to None.

    Returns:
        tuple of three lists of arrays:
            - event_time_NP (list of np.array): Times corresponding to each isolated event.
            - event_data_NP (list of np.array): Signal data for each isolated event.
            - event_start_end_idx (list of list): Starting and ending indices for each event.
    
    Notes:
        - Events are centered around peaks or troughs, adjusted by defined widths and extended by a 1000-point window.
        - Events are optionally merged or adjusted based on proximity to detected troughs.
        - If no corresponding trough is found for a peak, the event is isolated based only on the peak's properties.
    """
    
    event_data_NP = []
    event_time_NP = []
    event_start_end_idx = []
    count = 0
    processed_peaks = set()
    
    mean_noise = np.mean(y_base)
    sd_noise = np.std(y_base)

    
    # Monophasic peaks
    if peaks_below is None:
        for peak, width in zip(peaks_above, properties_above['widths']):
            # if the next peak along is in peaks_below, then contatenate the two peaks, then skip the peak in peaks_below
            # "peak" is the index of the peak in peaks_above
            start = max(0, peak - width // 2) - 300
            end = min(len(y_base), peak + width // 2) + 300
            
            if (end-start) < 1000: # makes sure error doesn't occur when you get an abnormally long event      
                event = y_base[int(start):int(end)]
                time = x_base[int(start):int(end)]
                event_data_NP.append(event)
                event_time_NP.append(time)
                event_start_end_idx.append([int(start), int(end)])
            else: 
                #take a 1000 point window around the peak if event abnormally long
                start = peak - 500
                end = peak + 500
                event = y_base[int(start):int(end)]
                time = x_base[int(start):int(end)]
                event_data_NP.append(event)
                event_time_NP.append(time)
                event_start_end_idx.append([int(start), int(end)])
    
    # Biphasic peaks
    else:
        
        for trough in peaks_below:
            peak = None # added to correct error
            previous_peak = find_previous_peak(trough, peaks_above)
            next_peak = find_next_peak(trough, peaks_above)
            
            # work out if trough is closer to previous or next peak
            # need check for the case where there is no previous or next peak
            
            if previous_peak is not None and next_peak is not None:
                if abs(trough - previous_peak) < abs(trough - next_peak):
                    peak = previous_peak
                    peak_width_start = properties_above['widths'][np.where(peaks_above == previous_peak)[0][0]]
                    peak_width_end = properties_below['left_ips'][np.where(peaks_below == trough)[0][0]]
                else:
                    peak = next_peak
                    peak_width_start = properties_below['right_ips'][np.where(peaks_below == trough)[0][0]]
                    peak_width_end = properties_above['widths'][np.where(peaks_above == next_peak)[0][0]]
            elif previous_peak is not None:
                # end of trace
                peak = previous_peak
                peak_width_start = properties_above['widths'][np.where(peaks_above == previous_peak)[0][0]]
                peak_width_end = properties_below['left_ips'][np.where(peaks_below == trough)[0][0]]
            elif next_peak is not None:
                # beginning of trace
                peak = next_peak
                peak_width_start = properties_below['right_ips'][np.where(peaks_below == trough)[0][0]]
                peak_width_end = properties_above['widths'][np.where(peaks_above == next_peak)[0][0]]
                

            if peak is not None and peak not in processed_peaks:
                # checks peak hasn't been counted twice
                if trough < peak:
                    start = peak_width_start  -300 # here peak_width_start is the left_ips of the peak below -> ips isnt a width, its a position so we dont need to add/takw away half from the trough position 
                    end = min(len(y_base), peak + peak_width_end // 2) +200 # End at the peak

                else:
                    start = max(0, peak - peak_width_start // 2) -200 # Start at the peak
                    end = + peak_width_end +300 # End at the trough

                event = y_base[int(start):int(end)]
                time = x_base[int(start):int(end)]
                
                
                if (end-start) < 1000:
                    event_data_NP.append((peak, event))
                    event_time_NP.append((peak, time))
                    event_start_end_idx.append((peak, [int(start), int(end)]))
                    count += 1
                
                else:
                    start = peak - 500
                    end = peak + 500
                    event = y_base[int(start):int(end)]
                    time = x_base[int(start):int(end)]
                    event_data_NP.append((peak, event))
                    event_time_NP.append((peak, time))
                    event_start_end_idx.append((peak, [int(start), int(end)]))
                    
                # add index to process peaks to keep track
                processed_peaks.add(peak)
                

        # Isolate peaks that don't have a corresponding trough
        for peak, width in zip(peaks_above, properties_above['widths']):
            if peak not in processed_peaks:
                start = max(0, peak - width // 2) -200
                end = min(len(y_base), peak + width // 2) +200
                event = y_base[int(start):int(end)]
                time = x_base[int(start):int(end)]
                
                if (end-start) < 1000:                    
                    event_data_NP.append((peak, event))
                    event_time_NP.append((peak, time))
                    event_start_end_idx.append((peak, [int(start), int(end)]))
                    count += 1
                    
                else:
                    start = peak - 500
                    end = peak + 500
                    event = y_base[int(start):int(end)]
                    time = x_base[int(start):int(end)]
                    event_data_NP.append((peak, event))
                    event_time_NP.append((peak, time))
                    event_start_end_idx.append((peak, [int(start), int(end)]))
                
        # sort the events by the index to ensure original order
        event_data_NP.sort(key=lambda x: x[0]) # sort
        event_data_NP = [np.array(data) for _, data in sorted(event_data_NP, key=lambda x: x[0])] # Remove index from event_data_NP and convert to list of arrays
        event_time_NP.sort(key=lambda x: x[0])
        event_time_NP = [np.array(time) for _, time in sorted(event_time_NP, key=lambda x: x[0])]
        event_start_end_idx = [idx for _, idx in sorted(event_start_end_idx, key=lambda x: x[0])]

    return event_time_NP, event_data_NP, event_start_end_idx





def event_isolation_even(x_base, y_base, peaks_above, properties_above, peaks_below = None, properties_below = None):

    event_data_NP = []
    event_time_NP = []

    for peak in peaks_above:

        start = peak - 500
        end = peak + 500
        event = y_base[int(start):int(end)]
        time = x_base[int(start):int(end)]
        event_data_NP.append(event)
        event_time_NP.append(time)
        
    smo = None
    
    return event_time_NP, event_data_NP, smo



def event_isolation_NRNS(x_base, y_base, peaks_above, NR = False, NP = False): 
    """
    Isolates events for signals that may contain a small resistive part, with specific adjustments for NR (Non-Resistive) and NP (Non-Peak) modes.

    Args:
        x_base (np.array): Array of time indices or equivalent sequential data points.
        y_base (np.array): Array of baseline-corrected signal values.
        peaks_above (np.array): Indices of peaks in the y_base data.
        NR (bool, optional): Specifies if the isolation should be tuned for Non-Resistive data. Defaults to False.
        NP (bool, optional): Specifies if the isolation should be tuned for Non-Peak data, affecting the padding. Defaults to False.

    Returns:
        tuple of lists of arrays:
            - event_time_NP (list of np.array): Times corresponding to each isolated event.
            - event_data_NP (list of np.array): Signal data for each isolated event.
    """
    event_data_NP = []
    event_time_NP = []

    
    if NR:
        for peak in peaks_above:
            start = peak - 1000
            end = peak + 1000
        
            event = y_base[int(start):int(end)]
            time = x_base[int(start):int(end)]
            event_data_NP.append(event)
            event_time_NP.append(time)
    
    else:
        mean_noise = np.mean(y_base)
        sd_noise = np.std(y_base)
        
        for peak in peaks_above:
            start = peak - 400 if NP else peak - 500
            end = peak + 400 if NP else peak + 500 
            
            event = y_base[int(start):int(end)]
            time = x_base[int(start):int(end)]
            
            # to avoid picking up 2 events at once and make sure same length as NR events
            if NP:
                time, event = pad_single_event(time, event, mean_noise, custom_length = 1000) 
            else:
                time, event = pad_single_event_smooth(time, event, mean_noise, sd_noise/7, custom_length = 2000) 
            

            event_data_NP.append(event)
            event_time_NP.append(time)
    

    return event_time_NP, event_data_NP





def lowpassfilter(signal, thresh=0.1, wavelet="coif4"):
    thresh = thresh * np.nanmax(signal)
    coeffs = pywt.wavedec(signal, wavelet, mode="per")

    # Apply thresholding
    coeffs[1:] = [pywt.threshold(i, value=thresh, mode="soft") for i in coeffs[1:]] # was soft

    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode="per")
    return reconstructed_signal, coeffs

def lowpassfilter_all(signal, thresh=0.1, wavelet="coif4"):
    thresh = thresh * np.nanmax(signal)
    coeffs = pywt.wavedec(signal, wavelet, mode="per")

    # Apply thresholding
    coeffs = [pywt.threshold(i, value=thresh, mode="soft") for i in coeffs] # was soft

    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode="per") # per should maintain length of signal but doesn't always
    
    if len(reconstructed_signal) > len(signal):
        reconstructed_signal = reconstructed_signal[:len(signal)]
    
    return reconstructed_signal, coeffs

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


# def load_to_event_data_nofeatures(path, resistive = False):
#     x, y, sma, y_corrected, y_base, x_base = importABF_movingavg(path, resistive) 
    
#     threshold, mean_noise, sd_noise = define_threshold(y_base, 12)
    
#     sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below = find_peaks_troughs(mean_noise, sd_noise, y_base, x_base, resistive)
    
#     if resistive:
#         event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above, peaks_below, properties_below)
#     else:
#         event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above)        
    
    
#     return event_time, event_data, smo, sd_threshold, sd_threshold_lower, mean_noise #, peaks_above, peaks_below

# def load_to_event_data_nofeatures(path, resistive = False, plot = False):
#     x, y, sma, y_corrected, y_base, x_base = importABF_movingavg(path, resistive) 
    
#     threshold, mean_noise, sd_noise = define_threshold(y_base, 12)
    
#     sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below = find_peaks_troughs(mean_noise, sd_noise, y_base, x_base, resistive, plot = plot)
    
#     if resistive:
#         event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above, peaks_below, properties_below)
#     else:
#         event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above)        
    
#     return event_time, event_data, smo, sd_threshold, sd_threshold_lower, mean_noise #, peaks_above, peaks_below

#load_to_event_data_nofeatures

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






def load_to_event_data_nofeatures_NRNS(path, resistive = False, small_resistive = False, NR = False, plot = False):
    """ allows analysis of nanorods and nanospheres data as well

    Returns:
        event_time, event_data, smo, sd_threshold, sd_threshold_lower, mean_noise
    """
    x, y, sma, y_corrected, y_base, x_base = importABF_movingavg(path, resistive) 
    
    threshold, mean_noise, sd_noise = define_threshold(y_base, 12) # only uses mean and sd, next func calcs uneven thresholds
    
    sd_threshold, sd_threshold_lower, peaks_above, properties_above, peaks_below, properties_below = find_peaks_troughs(mean_noise, sd_noise, y_base, x_base, resistive, plot = plot)
    
    if resistive and small_resistive:
        event_time, event_data = event_isolation_NRNS(x_base, y_base, peaks_above, NR = NR)
        smo = None
    elif resistive:
        event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above, peaks_below, properties_below)
    else:
        event_time, event_data, smo = event_isolation(x_base, y_base, peaks_above, properties_above)
    
    return event_time, event_data, smo, sd_threshold, sd_threshold_lower, mean_noise






#conventional DWT application
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




# def pad_event(event_time, event_data, mean_noise):
#     max_length = max([len(i) for i in event_data])
#     data_padded = [np.pad(i, (0, max_length - len(i)), 'constant', constant_values=mean_noise) for i in event_data]

#     time_padded = []
#     for i in event_time:
#         diff = max_length - len(i)
#         if diff > 0:
#             # Calculate the increment based on the current time array
#             increment = i[1] - i[0]
#             # Create an array of padded values
#             pad_values = i[-1] + np.arange(1, diff + 1) * increment
#             # Append the pad values to the current time array
#             i_padded = np.append(i, pad_values)
#         else:
#             i_padded = i

#         time_padded.append(i_padded)
        
#     return time_padded, data_padded


# event padding
def pad_event(event_time, event_data, mean_noise, custom_length = None):
    """
    Pads event data and time sequences to ensure uniform length across all events, filling with the specified noise mean or calculated values.

    Args:
        event_time (list of np.array): List of arrays containing time points for each event.
        event_data (list of np.array): List of arrays containing data points for each event.
        mean_noise (float): Mean noise level used to fill padding for data where needed.
        custom_length (int, optional): Specific length to pad all events to. If not provided, pads to the length of the longest event.

    Returns:
        tuple: A tuple containing two lists:
            - time_padded (list of np.array): Time arrays padded to uniform length.
            - data_padded (list of np.array): Data arrays padded to uniform length.
    """
    
    if custom_length:
        max_length = custom_length
    else:
        max_length = max(len(i) for i in event_data)
    
    # pad the current data with mean noise
    data_padded = [np.pad(i, ((max_length - len(i) + 1) // 2, (max_length - len(i)) // 2), 'constant', constant_values=mean_noise) for i in event_data]

    # pad time data
    time_padded = []
    for i in event_time:
        current_length = len(i)
        if current_length == 0:
            i_padded = np.full(max_length, mean_noise)  # Assuming mean_noise is a reasonable fill value
        elif current_length < max_length:
            # Calculate the padding size for the start and end
            start_padding_size = (max_length - current_length + 1) // 2
            end_padding_size = (max_length - current_length) // 2

            increment = (i[-1] - i[-2]) if current_length > 1 else 2e-6  # Default increment to 1 if not computable

            start_pad_values = i[0] - np.arange(start_padding_size, 0, -1) * increment
            end_pad_values = i[-1] + np.arange(1, end_padding_size + 1) * increment

            i_padded = np.concatenate([start_pad_values, i, end_pad_values])
        else:
            i_padded = i

        time_padded.append(i_padded)

    return time_padded, data_padded




def pad_single_event(event_time, event_data, mean_noise, custom_length):
    # Determine the length to pad to
    max_length = custom_length
    
    # Pad the event data
    data_padding = ((max_length - len(event_data) + 1) // 2, (max_length - len(event_data)) // 2)
    data_padded = np.pad(event_data, data_padding, 'constant', constant_values=mean_noise)

    # Pad the event time
    if len(event_time) < max_length:
        start_padding_size = (max_length - len(event_time) + 1) // 2
        end_padding_size = (max_length - len(event_time)) // 2

        increment = (event_time[-1] - event_time[-2]) if len(event_time) > 1 else 2e-6
        start_pad_values = event_time[0] - np.arange(start_padding_size, 0, -1) * increment
        end_pad_values = event_time[-1] + np.arange(1, end_padding_size + 1) * increment

        time_padded = np.concatenate([start_pad_values, event_time, end_pad_values])
    else:
        time_padded = event_time

    return time_padded, data_padded



def pad_single_event_smooth(event_time, event_data, mean_noise, sd_noise, custom_length, transition_length=200, noise_frequency=5):
    
    if custom_length == len(event_data):
        return event_time, event_data
    
    # Ensure that transition_length is not greater than the custom length
    transition_length = min(transition_length, custom_length // 2)

    # Calculate the padding size on each side of the event
    padding_length = custom_length - len(event_data)
    padding_each_side = padding_length // 2

    # Function to create a transition with a predefined noise level
    def transition_with_noise(start_value, end_value, length, noise_level, noise_frequency = noise_frequency):
        length = int(length)
        transition = np.linspace(start_value, end_value, length)
        # Apply noise at specified frequency, otherwise transition is very fuzzy
        noise_indices = np.arange(0, length, noise_frequency)
        noise = np.zeros(length)
        noise[noise_indices] = np.random.normal(0, noise_level, len(noise_indices))
        return transition + noise

    # Create transition padding
    start_transition = transition_with_noise(mean_noise, event_data[0], transition_length, sd_noise)
    end_transition = transition_with_noise(event_data[-1], mean_noise, transition_length, sd_noise)

    # Create full padding with transition
    start_padding = np.concatenate([np.full(padding_each_side - transition_length, mean_noise), start_transition])
    end_padding = np.concatenate([end_transition, np.full(padding_each_side + padding_length % 2 - transition_length, mean_noise)])

    data_padded = np.concatenate([start_padding, event_data, end_padding])

    # Pad the event time
    increment = (event_time[-1] - event_time[-2]) if len(event_time) > 1 else 2e-6
    start_pad_time = np.linspace(event_time[0] - padding_each_side * increment, event_time[0] - increment, padding_each_side)
    end_pad_time = np.linspace(event_time[-1] + increment, event_time[-1] + (padding_each_side + padding_length % 2) * increment, padding_each_side + padding_length % 2)

    time_padded = np.concatenate([start_pad_time, event_time, end_pad_time])

    return time_padded, data_padded







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





def unpickle(pickle_file):
    pickle_file = "/Users/joehart/Desktop/1_Imperial/Year 4/MSci project/Python_nanopores/files_nanopores_pkl/" + pickle_file
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)
    
def save_with_pickle(file_name, data):
    file_name = "/Users/joehart/Desktop/1_Imperial/Year 4/MSci project/Python_nanopores/files_nanopores_pkl/" + file_name
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
        
        
def DWT_and_features(event_time, event_data, mean_noise, sd_threshold, sd_threshold_lower, NP_size, lowpass_all = False):
    
    DWT_rec, all_coeffs = zip(*[wavelet_transform_func(signal) for signal in event_data])
    DWT_rec, all_coeffs = list(DWT_rec), list(all_coeffs)
    event_time_padded, DWT_rec_padded = pad_event(event_time, DWT_rec, mean_noise)
    features_df, features_list =  find_features(event_time_padded, DWT_rec_padded, mean_noise, all_coeffs, sd_threshold, sd_threshold_lower)
    if NP_size == 5:
        labels = list(np.zeros(len(features_list)))
    else:
        labels = list(np.ones(len(features_list)))
    
    return DWT_rec, event_time_padded, DWT_rec_padded, features_df, features_list, labels, all_coeffs



# apply thresholding to all levels of the wavelet transform
def DWT_and_features_thresh(event_time, event_data, mean_noise, sd_threshold, sd_threshold_lower, NP_size, wavelet, threshold):
    DWT_rec, all_coeffs = zip(*[wavelet_transform_func(signal, wavelet=wavelet, thresh = threshold) for signal in event_data])
    DWT_rec, all_coeffs = list(DWT_rec), list(all_coeffs)
    
    event_time_padded, DWT_rec_padded = pad_event(event_time, DWT_rec, mean_noise)
    features_df, features_list = find_features(event_time_padded, DWT_rec_padded, mean_noise, all_coeffs, sd_threshold, sd_threshold_lower)
    labels = [0 if NP_size == 5 else 1] * len(features_list)
    return DWT_rec, event_time_padded, DWT_rec_padded, features_df, features_list, labels, all_coeffs



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


# extracting features after DWT of whole trace + event isolation
def features_DWT_whole(event_time, event_data, mean_noise, sd_threshold, sd_threshold_lower, all_coeffs, NP_size = None):
    event_time_padded, event_data_padded = pad_event(event_time, event_data, mean_noise)
    features_df, features_list =  find_features(event_time_padded, event_data_padded, mean_noise, all_coeffs, sd_threshold, sd_threshold_lower)
    
    if NP_size == 5:
        labels = list(np.zeros(len(features_list)))
    elif NP_size is None:
        print("NP_size not specified")
        labels = None
    else:
        labels = list(np.ones(len(features_list)))
    
    return event_time_padded, event_data_padded, features_df, features_list, labels




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






def hyperparam_op(model, type, X, y, no_train_test = False):
    """
    Conducts hyperparameter optimization for different types of classifiers using specified search strategies.

    Args:
        model (str): Classifier type to use ('DT' for Decision Tree, 'RF' for Random Forest, 'XG' for XGBoost, 'SVM' for Support Vector Machine).
        type (str): Type of search to perform ('random', 'grid', 'sobol', 'bayes').
        X (array-like): Feature array.
        y (array-like): Label array.
        no_train_test (bool, optional): If True, skips splitting the dataset into training and testing subsets. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - X_train (array-like): Training data subset or the entire dataset if no_train_test is True.
            - X_test (array-like or None): Testing data subset or None if no_train_test is True.
            - y_train (array-like): Training labels or the entire label set if no_train_test is True.
            - y_test (array-like or None): Testing labels or None if no_train_test is True.
            - y_pred (array-like): Predicted labels for the training set.
            - search (estimator): Trained model.
            - best_params (dict): Best parameters found during the optimization.
    """
    
    if no_train_test:
        # already have the train test split
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        # stratified split to ensure equal number of each class in train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # initialise k fold cross validation so that its the same splits for every test --> fair test and accuracy comparison
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # which model
    if model == "DT":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('DT', DecisionTreeClassifier(random_state=42))
        ])
        # model parameter space
        params = {"DT__max_depth": [3, None],
              "DT__max_features": randint(1, 9),
              "DT__min_samples_leaf": randint(1, 9),
              "DT__criterion": ["gini", "entropy"]}

    elif model == "RF":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('RF', RandomForestClassifier(random_state=42))
        ])
        
        params = {
            'RF__bootstrap': [True, False],
            'RF__n_estimators': [10, 17, 25, 33, 41, 48, 56, 64, 72, 80],
            'RF__max_depth': randint(3, 20),
            'RF__min_samples_split': randint(2, 11),
            'RF__min_samples_leaf': randint(1, 11),
            'RF__max_features': ['sqrt', None]
        }
        
    elif model == "XG":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('XG', XGBClassifier(random_state=42))
        ])
        
        params = {
            'XG__n_estimators': randint(100, 1000),
            'XG__max_depth': randint(3, 20),
            'XG__learning_rate': expon(scale=0.1),
            'XG__subsample': [0.5, 0.75, 1.0],
            'XG__min_child_weight' : [ 1, 3, 5, 7 ],
            'XG__gamma': [ 0.0, 0.05, 0.1, 0.15, 0.2 , 0.3, 0.4 ],
            'XG__colsample_bytree': [ 0.3, 0.4, 0.5 , 0.6, 0.7 , 0.8, 0.9, 1.0 ]} #lambda and alpha for regularisation
        
    elif model == "SVM":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('SVM', SVC(random_state=42))
        ])
        
        params = {
            'SVM__C': expon(scale=100),
            'SVM__kernel': ['linear', 'rbf'],
            'SVM__gamma': ['scale', 'auto']}
        
    # which optimisaiton technique
    if type == "random":
        search = RandomizedSearchCV(
            clf_pipeline,
            params,
            n_iter=60,
            cv=kf,
            scoring='accuracy',
            random_state=42
        )
    elif type == "grid":
        search = GridSearchCV(
            clf_pipeline,
            params,
            cv=kf,
            scoring='accuracy',
        )
    elif type == "sobol":
        search = BayesSearchCV( 
            clf_pipeline,
            params,
            cv=kf,
            scoring='accuracy',
            n_iter=60,
            random_state=42, 
            n_initial_points=10 # in scikit-optimize, the Sobol sequence is used by default when initial points specified
        )
    elif type == "bayes":
        search = BayesSearchCV(
            clf_pipeline,
            params,
            cv=kf,
            scoring='accuracy',
            n_iter=60,
            random_state=42,
            n_jobs=-1
        )
    
    # fit training data
    search.fit(X_train, y_train)
    accuracies = search.best_score_
    y_pred = cross_val_predict(search.best_estimator_, X_train, y_train, cv=kf, method = "predict")

    average_accuracy = search.cv_results_['mean_test_score'][search.best_index_]

    best_params = search.best_params_
    print(f"{model} best params using {type}:", best_params)
    print("Average CV accuracy:",average_accuracy, "\u00B1", np.std(accuracies))
    print(f"Average CV accuracy: {average_accuracy:.10f} ± {np.std(accuracies):.10f}")
    
    return X_train, X_test, y_train, y_test, y_pred, search, best_params




def df_flatten(df):
    new_data_cols = {}
    for column in df.columns:
        combined_list = []
        for sublist in df[column]:
            combined_list.extend(sublist)
        new_data_cols[column] = [combined_list]

    df_flat = pd.DataFrame(new_data_cols)
    
    return df_flat



def centre_peak(time, data, peak_loc):
    """ 
    Args:
        time, data: list of arrays

    Returns:
        centered_time, centered_data
    """
    centered_data = []
    centered_time = []
    for i in range(len(data)):
        peak = np.argmax(data[i])
        shift = peak_loc - peak
        centered_data.append(np.roll(data[i], shift))
        centered_time.append(np.roll(time[i], shift))
    
    return centered_time, centered_data




##### metrics

# def apply_RMSE(df, df_noDWT_event_data, thresh):
#     RMSE_dic = {}
#     for col in df.columns:
#         rmse_sum = 0
#         for i in range(len(df[col][1])):
#             sequence1 = np.array(df_noDWT_event_data["data"][0][i])
#             sequence2 = np.array(df[col][1][i])
            
#             # Calculate Euclidean distance for the sequences
#             rmse = np.sqrt(np.mean((sequence1 - sequence2)**2))
#             rmse_sum += rmse
        
#         RMSE_dic[col] = rmse_sum / len(df[col][1]) 
        
#     return pd.DataFrame(RMSE_dic, index=[f"thresh_{thresh}"]).T


# def smoothness(df, df_noDWT_event_data, thresh):
#     """_summary_: Calculates smoothness of the DWT signal compared to the raw signal
#                   by subtracting the raw signal from the DWT signal and taking the standard deviation

#     Returns:
#         _type_: df
#     """
#     smoothness_dic = {}
#     for col in df.columns:
#         sum_sum = 0
#         for i in range(100,200):
#             smooth = df_noDWT_event_data["data"][0][i] - df[col][1][i]
#             sum_sum += np.std(smooth)
            
#         smoothness_dic[col] = sum_sum / 100
            
#     return pd.DataFrame(smoothness_dic, index = [f"thresh_{thresh}"]).T


# def metric(loss_XG_all_thresh_coeff2, rmse_df_norm, smoothness_all_norm, a, b, c):
#     # alpha controls weighting towards accuracy
#     metric_df = pd.DataFrame()
#     for col in rmse_df_norm.columns:

#         metric_df[col] = a * rmse_df_norm[col] + b * loss_XG_all_thresh_coeff2[col] + c * smoothness_all_norm[col]
    
#     return metric_df

def time_between_events(event_times):
    t_between = []
    for i in range(len(event_times) - 1):
        t_diff = event_times[i+1][0] - event_times[i][-1]
        t_between.append(t_diff)
    return t_between

def time_between_multirun(event_times):
    t_between = []
    for i in range(len(event_times) - 1):
        t_diff = event_times[i+1][0] - event_times[i][-1]
        if t_diff > 0:
            t_between.append(t_diff)
    return t_between