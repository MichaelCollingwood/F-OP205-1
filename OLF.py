""" VSet_OLF """

import random
import librosa
from scipy.signal import spectrogram, find_peaks
from tqdm import tqdm
import numpy as np

from DataManipulation import ButterLowpassFilter
from OldFindingPeaks import FilterPeaks
from TrainingInputData import GetNormativeManualScoring, TrimUtterance, GetInflexionTimes, ThinInflexionTimes
from ErrorObject import ErrorObject




def GetVSet_OLF(nperseg, cutoffSmPCA, thin_factor, window, plot_process_eg=False, plot_error=False):
    print("GetVSet_OLF")
    """
    --------------       -------------------------       ------------------
    | Utterances |  ==>  | Optimised Linear Func |  ==>  | Validated Sets |
    --------------       -------------------------       ------------------
    """
    
    
    """ Set Parameters """
    nMS = GetNormativeManualScoring()


    theta_ft = nperseg
    HV2 = cutoffSmPCA   # cutoff smoothing PCA plot
    T1=-0.8             # fixed spacing threshold
    T2=-0.6             # dynamic spacing threshold
    T3=-0.98            # fixed amplitude threshold
    T4=-0.1             # dynamic amplitude threshold 
    T5=0.21             # percentage components to consider
    C1=0.978            # cutoff smoothing average peak height
    C2=8.5              # cutoff smoothing average peak spacing
    theta = {'fPCA':[HV2, T1,T2,T3,T4,T5,C1,C2],'ft':theta_ft}
    
    
    """ Evaluate Per Utterance """
    errorObject = ErrorObject()
    vset_pa = {}; vset_ta = {}; vset_ka = {}; vset_pataka = {}; syllLengths = np.array([])
    
    keys =  list(nMS.keys())
    random.shuffle(keys)
    
    for key in tqdm(keys):
        manuallyScoredData = nMS[key]
        if (manuallyScoredData[0] == 'pa ta ka'):
            continue
        
        signal, rate = librosa.load("../audio_data/pataka-norms-audio/"+key+".wav", sr=None)
        signal = TrimUtterance(signal, nMS[key][1], nMS[key][2]) 
        F, times, Sxx = spectrogram(signal, fs=16000, window="hamming", nperseg=theta['ft']); fps = 1/(times[1]-times[0])
        
        y, locations = fPCA(Sxx, theta['fPCA'])
        inflexion_times = GetInflexionTimes(y, locations, times)
        thined_inflexion_times = ThinInflexionTimes(inflexion_times, thin_factor)
        
        errorObject.IncError(manuallyScoredData[0], len(locations), manuallyScoredData[3])
        
        if (len(locations) == manuallyScoredData[3]):
            if (manuallyScoredData[0] == 'pa pa pa'):
                vset_pa[key] = thined_inflexion_times
            if (manuallyScoredData[0] == 'ta ta ta'):
                vset_ta[key] = thined_inflexion_times
            if (manuallyScoredData[0] == 'ka ka ka'):
                vset_ka[key] = thined_inflexion_times
            if (manuallyScoredData[0] == 'pa ta ka'):
                vset_pataka[key] = thined_inflexion_times
                
            syllLengths = np.concatenate((syllLengths, GetDurations(ThinInflexionTimes(inflexion_times, -0.4))))
        
    errorObject.Normalise()
    
    print((syllLengths * fps)[:50])
    llambda_samples = list(syllLengths * fps / window)
    
    """ Performance Analysis """
    errorObject.Print()
    errorObject.Plot()
    
    return [vset_pa, vset_ta, vset_ka, vset_pataka], llambda_samples



def fPCA(X, theta_fPCA):
    [HV2, T1,T2,T3,T4,T5,C1,C2] = theta_fPCA
    
    U,s,V = np.linalg.svd(X.T)
    Y = np.zeros(X.shape[1])
    for i in range(int(len(V)*T5)):
        Xi = np.matmul(X.T, V[i])
        Xi /= np.sum(Xi)
        Y += s[i]*abs(Xi)
           
    y = ButterLowpassFilter(Y, HV2)
        
    peaks, peakProperties = find_peaks(y, height=0)
    filteredPeaks = FilterPeaks(y, peaks, peakProperties, T1, T2, T3, T4, C1, C2)
    
    return y, filteredPeaks

def GetDurations(intervals):
    durations = []
    
    for interval in intervals:
        durations.append(interval[1]-interval[0])
        
    return np.array(durations)