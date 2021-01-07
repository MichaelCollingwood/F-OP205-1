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
from TrainingInputData import GetNormativeManualScoring, ManuallyLabelPeaks
from FeatureExtractor import FeatureExtractor
from InputFormatting import ImageToTorchFormat
from DataStorage import OpenCache_llambda
from DataManipulation import ButterLowpassFilter
from PeaksMatrix import GetPeakVector
from Display import PlotSpecAndFunc


# BuildModels()

syll = 'pa'; window = 20; step = 1; nperseg = 100
model = Net()
model.load_state_dict(torch.load("CNN_models/model_{}_n{}w{}".format(syll,nperseg,window)))
model.eval()

    
def DemonstratePerformance(syll, window, step, nperseg):
    """
    
       Input
    o  PreProcess
    o  Feature Extraction
    
       ----------------------------
    o  | CNN: Probability of syll | time warping committee
       ---------------------------- weighted by chance
    
       -----------------------------
       | Smooth Envelope           |
       | Find Peaks                |
    o  | Feature Extraction: Peaks |
    o  | CNN: Prob peak is syll    |
       | Filter Peaks              |
       |                           |
    o  | Macro Matrix              |
    o  | CNN: Accuracy of model    | smoothing committee
       ----------------------------- weighted by CNN Accuracy * Prior
    
       Expected Rate & Expected Accuracy
      
    """
    
    """ Input """
    nMS = GetNormativeManualScoring()
    keys = [key for key in nMS.keys() if (nMS[key][0][:2] == syll) and (nMS[key][0] != "pa ta ka")]
    random.shuffle(keys)
    
    peakMatrix = []; output_p = []
    for key in keys[:1]:

        signal, rate = librosa.load("../audio_data/pataka-norms-audio/"+key+".wav")
        
        llambda_samples = OpenCache_llambda(nperseg)
        np.random.shuffle(llambda_samples)
        
        Times = np.array([]); Ys = np.array([])
        for llambda in tqdm(llambda_samples[:400]):
            extractor = FeatureExtractor(signal, rate)
            extractor.GetSpectrogram(nperseg)
            extractor.SetFeatures("spectrogram")
            
            # print(llambda)
            extractor.Window(int(llambda*window), step)
            extractor.SqueezeWindows(window)
            times, X = extractor.features
            X = ImageToTorchFormat(X)
    
            """ CNN Envelope """        
            with torch.no_grad():
                output = model(X)
            output = np.array(output[:,1])
            
            Times = np.concatenate((Times, times))
            Ys = np.concatenate((Ys, np.exp(output)))
        
        Ys = Ys[np.argsort(Times)]
        times = Times[np.argsort(Times)]
        y = ButterLowpassFilter(Ys, 0.006)
        y -= np.min(y)
        y /= np.max(y)
        
        """ Peaks """
        peaks, peakProperties = find_peaks(y, height=0)
        peaksVectors = GetPeakVector(peaks, y, times)
        
        
        for i, peak in enumerate(peaks):
            peakMatrix.append(peaksVectors[i])
                
            if (data_collect):
                PlotSpecAndFunc(extractor.spectrogram[0], extractor.spectrogram[1], extractor.spectrogram[2], times, y, np.array([peak]), nMS[key])
                
                if (input('is syllable?[y/]') == 'y'):
                    output_p.append(1)
                else:
                    output_p.append(0)
                    
        if (not data_collect):
            # predict
            model_p = Net_p()
            model_p.load_state_dict(torch.load("CNN_models/peaksCNN"))
            model_p.eval()
            
            pred = np.array([0])
            while (np.min(pred) == 0):
                peakMatrix = [row for i, row in enumerate(peakMatrix) if output[i] > 0.5]
                X = np.array([np.array(row) for row in peakMatrix])
                X = X.reshape(X.shape[0], 1, X.shape[1])
                X = torch.from_numpy(X)
                X = X.type(torch.FloatTensor)
            
                with torch.no_grad():
                    output = model_p(X)
                output = np.array(output[:,1])
                plt.plot(output); plt.title("Peak Syll-Likelihood"); plt.show()
                pred = np.where(output > 0.5, 1, 0)
                plt.plot(pred); plt.title("Peak Syll-Class"); plt.show()
                
                peaks = np.array([peak for i, peak in enumerate(peaks) if output[i] > 0.5])
                PlotSpecAndFunc(extractor.spectrogram[0], extractor.spectrogram[1], extractor.spectrogram[2], times, y, peaks, nMS[key])
    
    if (data_collect):
        if (os.path.exists("output.json")):
            with open("output.json", "r") as read_file:
                Y = json.load(read_file)
            for element in output_p:
                Y.append(element)
            with open("output.json", "w") as fp:
                json.dump(Y, fp)
        else:
            with open("output.json", "w") as fp:
                json.dump(output_p, fp)

        if (os.path.exists("peakMatrix.json")):
            with open("peakMatrix.json", "r") as read_file:
                X = json.load(read_file)
            for row in peakMatrix:
                X.append(row)
            with open("peakMatrix.json", "w") as fp:
                json.dump(X, fp)
        else:
            with open("peakMatrix.json", "w") as fp:
                json.dump(peakMatrix, fp)
        
        # train 1D CNN and save it
        BuildModels_p()        
   
ManuallyLabelPeaks(model, syll, window, step, nperseg)
DemonstratePerformance(syll, window, step, nperseg)
