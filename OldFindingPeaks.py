# FINDING PEAKS
import numpy as np
from scipy.signal import find_peaks

from DataManipulation import ButterLowpassFilter




def EliminateSmallSpacingSC(peaks, T1):
    # ELIMINATE SMALL SPACING STRICT CRITERIA
    
    peaksDistances = np.array([peaks[i+1] - peaks[i] for i in range(len(peaks)-1)])
    peaksDistancesAvg = np.average(peaksDistances)
    newPeaks = [peaks[0]]
    
    ignoreFlag = False
    for j, peaksDistance in enumerate(peaksDistances):
        if (ignoreFlag):
            newPeaks.append(peaks[j+1])
            ignoreFlag = False
            continue
        
        if (peaksDistance > (1+T1)*peaksDistancesAvg):
            newPeaks.append(peaks[j+1])
        else:
            ignoreFlag = True
            
    return np.array(newPeaks)

def EliminateSmallSpacingRC(peaks, T2, C3):
    # ELIMINATE SMALL SPACING RELATIVE CRITERIA
    
    peaksDistances = np.array([peaks[i+1] - peaks[i] for i in range(len(peaks)-1)])
    peaksDistancesAvg = ButterLowpassFilter(peaksDistances, cutoff=C3)
    newPeaks = [peaks[0]]
    
    ignoreFlag = False
    for k, peaksDistance in enumerate(peaksDistances):
        if (ignoreFlag):
            newPeaks.append(peaks[k+1])
            ignoreFlag = False
            continue
        
        if (peaksDistance > (1+T2)*peaksDistancesAvg[k]):
            newPeaks.append(peaks[k+1])
        else:
            ignoreFlag = True
    
    return np.array(newPeaks)

def FilterPeaksBySpacing(peaks, T1, T2, C3):
    
    peaks = EliminateSmallSpacingSC(peaks, T1)
    peaks = EliminateSmallSpacingRC(peaks, T2, C3)
    
    return peaks




def EliminateSmallAmplitudesSC(y, peakProperties, T3):
    # ELIMINATE REL LOW PEAKS STRICT CRITERIA
    
    if (T3 != None):
        avgPeakHeight = np.average(peakProperties["peak_heights"])
        peaks, peakProperties = find_peaks(y, (1+T3)*avgPeakHeight)
    else:
        peaks, peakProperties = find_peaks(y, height=np.min(y))
    
    return peaks, peakProperties

def EliminateSmallAmplitudesRC(y, peaks, peakProperties, T4, C2):
    # ELIMINATE REL LOW PEAKS RELATIVE CRITERIA
    
    if (T4 != None):
        peakHeights = peakProperties["peak_heights"]
        avgHeight = ButterLowpassFilter(y, C2)
        peaks = [peaks[i] for i in range(len(peaks)) if (peakHeights[i] > (1+T4)*avgHeight[peaks[i]])]
    
    return peaks

def FilterPeaksByAmplitude(y, T3, T4, C2, peakProperties):
    
    peaks, peakProperties = EliminateSmallAmplitudesSC(y, peakProperties, T3)
    peaks = EliminateSmallAmplitudesRC(y, peaks, peakProperties, T4, C2)
    
    return peaks




def FilterPeaks(y, peaks, peakProperties, T1, T2, T3, T4, C2, C3):
    
    if (len(peaks) > 9):
        peaks = FilterPeaksByAmplitude(y, T3, T4, C2, peakProperties)
    if (len(peaks) > 9):
        peaks = FilterPeaksBySpacing(peaks, T1, T2, C3)
        
    #else:
    #    print("ERROR: PEAKS NOT FILTERED")
    
    return peaks




def FindPeakFrequencies(X, locations, width = 1):
    
    Xspecific = []
    for location in locations:
        Xspecific.append(X.T[location])
        for i in range(width):
            if ((location-i >= 0)&(location+i < len(X.T))):
                Xspecific.append(X.T[location-i])
                Xspecific.append(X.T[location+i])
                
    return np.array(Xspecific).T