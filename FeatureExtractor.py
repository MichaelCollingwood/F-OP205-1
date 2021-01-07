""" Feature Extractor """

from math import floor, ceil
import numpy as np
from scipy.signal import spectrogram

class FeatureExtractor():
    def __init__(self, signal, rate):
        self.signal = signal
        self.rate = rate
        
    def GetSpectrogram(self, nperseg):
        self.spectrogram = spectrogram(self.signal, fs=self.rate, window="hamming", nperseg=nperseg)
        
    def KPCA(self, n):
        U,s,V = np.linalg.svd(self.features[1].T)
        Y=[]
        for i in range(n):
            Y.append(np.matmul(self.features[1].T, V[i]))
        self.KPCA = (self.features[0], np.array(Y))
        
    def SetFeatures(self, featureType):
        # features = [times, 2D features arr]
        
        if (featureType == "spectrogram"):
            self.features = (self.spectrogram[1], self.spectrogram[2])
            
        if (featureType == "KPCA"):
            self.features = self.KPCA

            
    def Window(self, window, step):
        times = self.features[0]
        features = self.features[1]
        
        idx = np.array(range(window//2,len(times)-window//2,step))
        X = [features[:,j-(window//2):j+(window//2)+1] for j in idx]
        times_new = times[idx]
        
        self.features = (times_new, np.array(X))
        
    def SqueezeWindows(self, window):
        X_new = []
        
        for Xi in self.features[1]:
            Xi_new = np.zeros((Xi.shape[0], window+1))
            
            n = 0; nprev = 0
            for j, Xij in enumerate(Xi.T):
                n += window/len(Xi[0])
                
                nprev_floor = floor(nprev); nprev_ceil = ceil(nprev)
                n_floor = floor(n); n_ceil = ceil(n)
                # print(nprev_floor, nprev, nprev_ceil, n_floor, n , n_ceil)
                # going from nprev => nprev_ceil => n_floor => n
                
                Xi_new[:,nprev_floor] += (nprev_ceil - nprev)*Xij
                
                for i in range(nprev_ceil, n_floor+1):
                    Xi_new[:,i] = Xij
                
                if (n - n_floor > 0.001): # bc round-off error
                    Xi_new[:,n_ceil] += (n - n_floor)*Xij
                
                nprev = n
            
            X_new.append(Xi_new)
        self.features = (self.features[0], np.array(X_new))