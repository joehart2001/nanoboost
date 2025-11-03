import numpy as np
import pyabf

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
