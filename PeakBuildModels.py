""" Build Peak Models """

import json
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from PeakModelArchitectures import Net_p, train_p

def BuildModels_p():
    """ Extract Cached Data """
    with open("output.json", "r") as read_file:
        Y = json.load(read_file)

    Y = np.array(Y); print("Y.shape: ", Y.shape)
    Y = torch.from_numpy(Y)
    Y = Y.type(torch.LongTensor)
    
    with open("peakMatrix.json", "r") as read_file:
        X = json.load(read_file)
       
    X = np.array([np.array(row) for row in X]); print("X.shape: ", X.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X = torch.from_numpy(X)
    X = X.type(torch.FloatTensor)
    
    """ Setup Model """
    model = Net_p()
    optimizer = Adam(model.parameters(), lr=0.04)
    criterion = CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    n_epochs = 50
    for epoch in range(n_epochs):
        model = train_p(epoch, model, optimizer, criterion, X, Y)
    
    """ Save Model """
    torch.save(model.state_dict(), "CNN_models/peaksCNN")