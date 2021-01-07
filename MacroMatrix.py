""" Macro Matrix """

import torch
import numpy as np
import matplotlib.pyplot as plt

from PeaksMatrix import GetPeakVector

def GetMacroMatrix(signal, Sxx, y, times, peaks):
    
    # rate
    rate = len(peaks)/(times[-1] - times[0])
    
    # if volume max-out
    print(max(signal))
    plt.plot(signal); plt.show()
    
    # singular values
    U,s,V = np.linalg.svd(Sxx.T)
    
    # mean & stdv of peak features
    peaksMatrix = GetPeakVector(peaks, y, times)
    meanPeakVector = np.average(peaksMatrix, axis = 0)
    stdvPeakVector = np.std(peaksMatrix, axis = 0)
    
    output = np.concatenate((np.array([rate]), s, meanPeakVector, stdvPeakVector))
    print(len(output))
    #output = output.reshape(, )
    
    return output