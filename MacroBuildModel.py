""" Build Macro CNN Model """

import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from scipy.signal import spectrogram
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ModelArchitectures import Net
from PeakModelArchitectures import Net_p
from BuildModels import BuildModels
from PeakBuildModels import BuildModels_p
from TrainingInputData import GetNormativeManualScoring, ManuallyLabelPeaks, GetLabels
from FeatureExtractor import FeatureExtractor
from InputFormatting import ImageToTorchFormat
from DataStorage import OpenCache_llambda
from DataManipulation import ButterLowpassFilter
from PeaksMatrix import GetPeakVector
from Display import PlotSpecAndFunc
from MacroMatrix import GetMacroMatrix
from DataStorage import Cache, OpenCache, Cache_llambda
from OLF import GetVSet_OLF
from ModelArchitectures import Net, train
from TrainInput import TrainInput
from TrainingInputData import GetNormativeManualScoring, TrimUtterance, GetInflexionTimes, ThinInflexionTimes
from MacroModelArchitecture import Net_m, train_m



def Predict_noSmoothingCommittee(key, cutoff):
    """
          Input
    opt   PreProcess
    
         ----------------------------
    opt  | Feature Extraction       |
    opt  | CNN: Probability of syll | time warping committee
         ---------------------------- weighted by chance
    
          Smooth Envelope           
          Find Peaks                
    opt   Feature Extraction: Peaks 
    opt   CNN: Prob peak is syll    
          Filter Peaks              
                                    
    opt   Macro Matrix              
    opt   CNN: Accuracy of model
    
          Rate & Accuracy
    """
    
    
    syll = 'pa'; window = 20; step = 1; nperseg = 100
    model = Net(); model.load_state_dict(torch.load("CNN_models/model_{}_n{}w{}".format(syll,nperseg,window))); model.eval()
    model_p = Net_p(); model_p.load_state_dict(torch.load("CNN_models/peaksCNN")); model_p.eval()
    

    """ Input """
    signal, rate = librosa.load("../audio_data/pataka-norms-audio/"+key+".wav")
    
    
    """ PreProcess """
    
    
    """ Time Warping Committee """
    llambda_samples = OpenCache_llambda()
    np.random.shuffle(llambda_samples)
    
    Times = np.array([]); Ys = np.array([])
    for llambda in tqdm(llambda_samples[:400]):
        
        
        """ Feature Extraction """
        extractor = FeatureExtractor(signal, rate)
        extractor.GetSpectrogram(nperseg)
        extractor.SetFeatures("spectrogram")
        
        extractor.Window(int(llambda*window), step)
        extractor.SqueezeWindows(window)
        times, X = extractor.features
        X = ImageToTorchFormat(X)


        """ CNN: Probability of syll """        
        with torch.no_grad():
            output = model(X)
        output = np.array(output[:,1])
        
        Times = np.concatenate((Times, times))
        Ys = np.concatenate((Ys, np.exp(output)))
    
    
    # contributions by committee
    outputs = {}
    for i, time in enumerate(times):
        if (time not in outputs):
            outputs[time] = [Ys[i]]
        else:
            outputs[time].append(Ys[i])
    Ys = []; times = []
    for time in outputs:
        times.append(time)
        Ys.append(np.average(np.array(outputs[time])))
    Ys = np.array(Ys)[np.argsort(Times)]
    times = np.array(Times)[np.argsort(Times)]
    
    
    """ Smooth Envelope """
    y = ButterLowpassFilter(Ys, cutoff)
    y -= np.min(y)
    y /= np.max(y)
    
    
    """ Find Peaks """
    peaks, peakProperties = find_peaks(y, height=0)
    
    
    """ Feature Extraction: Peaks """
    peaksVectors = GetPeakVector(peaks, y, times)
    peakMatrix = []
    for i, peak in enumerate(peaks):
        peakMatrix.append(peaksVectors[i])


    """ Filter Peaks """
    pred = np.array([0])
    while (np.min(pred) == 0):
        X = np.array([np.array(row) for row in X])
        X = X.reshape(X.shape[0], 1, X.shape[1])
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        with torch.no_grad():
            output = model_p(X)
        output = np.array(output[:,1])
        
        plt.plot(output); plt.title("Peak Syll-Likelihood"); plt.show()
        
        peaks = np.array([peak for i, peak in enumerate(peaks) if output[i] > 0.5])
        X = np.array([xi for i, xi in enumerate(X) if output[i] > 0.5])
        PlotSpecAndFunc(extractor.spectrogram[0], extractor.spectrogram[1], extractor.spectrogram[2], times, y, peaks, GetLabels(key)) 
        
    rate = len(peaks) / (times[-1] - times[0])
        
    return signal, extractor.spectrogram[2], y, times, peaks, rate

def BuildModels_m():
        
    """ Setup Model """
    model = Net_m()
    optimizer = Adam(model.parameters(), lr=0.04)
    criterion = CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    nMS = GetNormativeManualScoring()
    
    Features = []; Errors = []
    for key in tqdm(nMS):
        
        if (nMS[key][0] == 'pa ta ka'):
            continue
        
        signal, r = librosa.load("../audio_data/pataka-norms-audio/"+key+".wav", sr=None)
        signal = TrimUtterance(signal, nMS[key][1], nMS[key][2]) 
        F, times, Sxx = spectrogram(signal, fs=16000, window="hamming", nperseg=100)
        
        for cutoff in np.linspace(0.006,0.01,10):
            signal, Sxx, y, times, peaks, rate = Predict_noSmoothingCommittee(key, cutoff)
            error = np.abs((rate - nMS[key][3]/(times[-1]-times[0])) / (nMS[key][3]/(times[-1]-times[0])))
            
            Features.append(GetMacroMatrix(signal, Sxx, y, times, peaks))
            Errors.append(error)
            
    Features = np.array(Features); Errors = np.array(Errors)
    Features = Features.reshape(1,1,Features.shape[-2],Features[-1])
    Features = torch.from_numpy(Features)
    Features = Features.type(torch.FloatTensor)
    
    Errors = torch.from_numpy(Errors)
    Errors = Errors.type(torch.LongTensor)
    
    n_epochs = 50
    for epoch in range(n_epochs):
        model = train_m(epoch, model, optimizer, criterion, Features, Errors)
    
    """ Save Model """
    torch.save(model.state_dict(), "CNN_models/macro_model")  