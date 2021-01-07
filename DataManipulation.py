# DATA ANALYTICS TOOLS
import numpy as np
from scipy.signal import butter,filtfilt


def ButterLowpassFilter(data, cutoff):
    if (len(data) <= 9):
        return data
    fs = 100      # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

