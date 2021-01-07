""" Error Object """

import matplotlib.pyplot as plt

class ErrorObject():
    def __init__ (self):
        self.errorPerUttRaw = {"pa pa pa": [0,0], "ta ta ta": [0,0], "ka ka ka": [0,0], "pa ta ka": [0,0]}
        self.errorPerUtt = {"pa pa pa": 0, "ta ta ta": 0, "ka ka ka": 0, "pa ta ka": 0}
        
        self.errorCounts = {}
        
    def IncError(self, uttType, predCount, trueCount):
        error = abs(predCount - trueCount)
        
        # update error metrics
        self.errorPerUttRaw[uttType] = [self.errorPerUttRaw[uttType][0] + error/trueCount, self.errorPerUttRaw[uttType][1] + 1]
        
        if (error in self.errorCounts):
            self.errorCounts[error] += 1
        else: 
            self.errorCounts[error] =  1
        
    def Normalise(self):
        N = 0
        for uttType in self.errorPerUttRaw:
            if (self.errorPerUttRaw[uttType][1] != 0):
                self.errorPerUtt[uttType] = self.errorPerUttRaw[uttType][0] / self.errorPerUttRaw[uttType][1]
                N += self.errorPerUttRaw[uttType][1]
        
        for errorCount in self.errorCounts:
            self.errorCounts[errorCount] /= N
            
    def Print(self):
        
        print("\nError Per Utterance Type:")
        for uttType in self.errorPerUtt:
            print("{}: {}%".format(uttType, round(self.errorPerUtt[uttType]*100, 2)))
            
    def Plot(self):
        
        X = list(self.errorCounts.keys())
        X.sort()
        Y = [self.errorCounts[key] if key in self.errorCounts else 0 for key in range(X[0]-1, X[-1])]

        plt.plot(range(X[0]-1, X[-1]), Y, marker='x', color='black')
        plt.ylabel('% of samples with error count = x')
        plt.xlabel('error count, x')
        plt.xlim(0); plt.ylim(0)
        plt.show()