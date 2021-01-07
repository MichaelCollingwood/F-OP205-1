""" Train Input """

import numpy as np
from tqdm import tqdm
import librosa
from sklearn.model_selection import train_test_split

from TrainingInputData import GetNormativeManualScoring, TrimUtterance, FindGroundTruth
from InputFormatting import ImageToTorchFormat, TargetToTorchFormat
from FeatureExtractor import FeatureExtractor

class TrainInput():
    def __init__(self, vset):
        self.vset = vset
    
    def GetInputArrs(self, window, step, nperseg):    
        """
        ----------------------       -------------       
        | Feature Extraction |  ==>  | Windowing |  =====  
        ----------------------       -------------       |    ----------------
                                                         ==>  | Input Arrays |
        -----------------       ---------------------    |    ----------------
        | Validated Set |  ==>  | Find Ground Truth |  ==  
        -----------------       ---------------------       
        """
        print("\nGetInputArrays:")
        
        nMS = GetNormativeManualScoring()
        
        self.X, self.Y = [], []
        for key in tqdm(self.vset):
            """ Get Labelled Input """
            signal, rate = librosa.load("../audio_data/pataka-norms-audio/"+key+".wav", sr=None)
            signal = TrimUtterance(signal, nMS[key][1], nMS[key][2]) 
            inflexion_times = np.array(self.vset[key])
            
            """ Extract Features """
            extractor = FeatureExtractor(signal, rate)
            extractor.GetSpectrogram(nperseg)
            extractor.SetFeatures("spectrogram")
            extractor.Window(window, step)
            times, x = extractor.features
            
            y = FindGroundTruth(times, inflexion_times)
            
            """ Accumulate Images and Labels """
            if (self.X == []):
                self.X = x
                self.Y = y
            else:
                self.X = np.concatenate((self.X, x))
                self.Y = np.concatenate((self.Y, y))     
    
    def Decimate(self, fraction):
        N = self.X.shape[0]
        
        idx = np.arange(0,N-1,1)
        np.random.shuffle(idx)
        idx = idx[:int(N*fraction)]
        
        self.train_x = self.X[idx]
        self.train_y = self.Y[idx]
        
    def Split(self, test_size=0.1):
        # train_x, val_x, train_y, val_y = train_test_split(self.train_x, self.train_y, test_size = 0.1)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y, test_size=0.1)
    
    def ConvertToTorch(self):
        self.train_x = ImageToTorchFormat(self.train_x)
        self.train_y = TargetToTorchFormat(self.train_y)
        
        self.val_x = ImageToTorchFormat(self.val_x)
        self.val_y = TargetToTorchFormat(self.val_y)
        
        return self.train_x, self.val_x, self.train_y, self.val_y
    