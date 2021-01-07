""" Input Formatting """

import torch



def ImageToTorchFormat(X):
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    X = torch.from_numpy(X)
    X = X.type(torch.FloatTensor)
    return X

def TargetToTorchFormat(Y):
    Y = torch.from_numpy(Y)
    Y = Y.type(torch.LongTensor)
    return Y