import numpy as np

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
            # check for the case where there is no previous or next peak
            
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
                    start = peak_width_start  -300 # here peak_width_start is the left_ips of the peak below -> ips isnt a width, its a position so we dont need to add/take away half from the trough position 
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
            
            event_data_NP.append(event)
            event_time_NP.append(time)
    

    return event_time_NP, event_data_NP



#event padding
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
            i_padded = np.full(max_length, mean_noise)
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