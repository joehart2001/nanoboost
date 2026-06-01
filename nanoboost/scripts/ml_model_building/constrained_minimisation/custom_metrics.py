import pandas as pd
import numpy as np
from nanoboost.scripts.utils.utils import unpickle


def apply_RMSE(df, df_noDWT_event_data, thresh):
    """
    Computes the Root Mean Square Error (RMSE) between raw and wavelet-transformed events, averaging over all events.

    Args:
        df (DataFrame): DataFrame the wavelet transformed events.
        df_noDWT_event_data (DataFrame): DataFrame containing the raw events.
        thresh (float): Threshold value

    Returns:
        DataFrame: A DataFrame with the average RMSE for each column in `df`, indexed by a threshold label.
    """
    RMSE_dic = {}
    for col in df.columns:
        rmse_sum = 0
        for i in range(len(df[col][1])): 
            sequence1 = np.array(df_noDWT_event_data["data"][0][i]) # raw
            sequence2 = np.array(df[col][1][i]) # smoothed
            
            # Calculate Euclidean distance for the sequences
            rmse = np.sqrt(np.mean((sequence1 - sequence2)**2))
            rmse_sum += rmse
        
        # Store the average Euclidean distance for the column
        RMSE_dic[col] = rmse_sum / len(df[col][1]) # average RMSE for the wavelet conditions
        
    return pd.DataFrame(RMSE_dic, index=[f"thresh_{thresh}"]).T



def smoothness(df, thresh):
    """
    Calculates a measure of smoothness for each event in a DataFrame based on the standard deviation of the residuals
    between the wavelet-transformed signal and the raw signal.

    Args:
        df (DataFrame): DataFrame containing raw and corresponding smoothed events for every wavelet
        thresh (float): Threshold
    Returns:
        DataFrame: A DataFrame with the smoothness value for each column in `df`, indexed by a threshold label.
    """
    smoothness_dic = {}
    for col in df.columns:
        sum_sum = 0
        for i in range(len(df[col][1])): #
            smooth = df["data"][0][i] - df[col][1][i] # difference
            sum_sum += np.std(smooth) # sd of difference proportional to smoothness
            
        smoothness_dic[col] = sum_sum / len(df[col][1])
            
    return pd.DataFrame(smoothness_dic, index = [f"thresh_{thresh}"]).T



def metric(accuracy_loss_df, rmse_df, smoothness_loss_df, a, b, c):
    """
    Computes a custom weighted evaluation metric from normalized RMSE, XGBoost loss, and smoothness scores for each wavelet.

    Args:
        rmse_df (DataFrame): Normalised RMSE scores for all ML models.
        smoothness_loss_df (DataFrame): Normalised smoothness scores for all ML models.
        accuracy_loss_df (DataFrame): Normalised accuracy loss for all ML models.
        a (float): Weight coefficient for the accuracy loss component of the metric.
        b (float): Weight coefficient for the RMSE.
        c (float): Weight coefficient for the smoothness loss.

    Returns:
        DataFrame: A DataFrame containing the computed metric for each ML model.
    """
    metric_df = pd.DataFrame()
    for col in rmse_df.columns:

        metric_df[col] = a * accuracy_loss_df[col] + b * rmse_df[col] + c * smoothness_loss_df[col]
    return metric_df