""" Demonstrate Performance """

import os
import torch
import random
import json
import librosa
from tqdm import tqdm
import numpy as np
from matplotlib.colors import LogNorm
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
from MacroBuildModel import BuildModels_m


# BuildModels()
# BuildModels_p()
BuildModels_m()

    
def Predict(key):
    """
          Input
    opt   PreProcess
    
         ----------------------------
    opt  | Feature Extraction       |
    opt  | CNN: Probability of syll | time warping committee
         ---------------------------- weighted by chance
    
         -----------------------------
         | Smooth Envelope           |
         | Find Peaks                |
    opt  | Feature Extraction: Peaks |
    opt  | CNN: Prob peak is syll    |
         | Filter Peaks              |
         |                           |
    opt  | Macro Matrix              |
    opt  | CNN: Accuracy of model    | smoothing committee
         ----------------------------- chosen by CNN Accuracy
    
          Expected Rate & Expected Accuracy
    """
    
    
    syll = 'pa'; window = 20; step = 1; nperseg = 100
    model = Net(); model.load_state_dict(torch.load("CNN_models/model_{}_n{}w{}".format(syll,nperseg,window))); model.eval()
    model_p = Net_p(); model_p.load_state_dict(torch.load("CNN_models/peaksCNN")); model_p.eval()
    model_m = Net_m(); model_m.load_state_dict(torch.load("CNN_models/macroCNN")); model_m.eval()
    

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
    
    
    """ Smoothing committee """
    weightings = {}
    for cutoff in tqdm(np.linspace(0.006,0.01,10)):
        
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
        
        
        """ Macro Matrix """
        macro = GetMacroMatrix(signal, X, y, times, peaks)
        macro = torch.from_numpy(macro)
        macro = macro.type(torch.FloatTensor)
        
        
        """ CNN: Accuracy of model """
        with torch.no_grad():
            output = model_m(macro)
        accuracy = np.array(output[:,1])
        
        weightings[rate] = accuracy
    
    
    """ Expected Rate & Expected Accuracy """
    bestRate = np.array(weightings.keys)[np.argmax(np.array(weightings.values))]
    maxAcc = np.array(weightings.values)[np.argmax(np.array(weightings.values))]
    
    return bestRate, maxAcc

# ManuallyLabelPeaks(model, syll, window, step, nperseg)
nMS = GetNormativeManualScoring()
Predict("eb3be048-de1a-4dd6-91a5-0cfd22607c6d")
