""" Build Models """

import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from DataStorage import Cache, OpenCache, Cache_llambda
from OLF import GetVSet_OLF
from ModelArchitectures import Net, train
from TrainInput import TrainInput

def BuildModels():
    
    """ Get Validated Sets """
    # Parameters
    window = 20
    step = 10
    
    nperseg = 100
    cutoffSmPCA = 3
    thin_factor = 0.8
    fraction = 0.4
    
    if (not os.path.exists("CNN_vset_n{}".format(nperseg))):
        [vset_pa, vset_ta, vset_ka, vset_pataka], llambda_samples = GetVSet_OLF(nperseg, cutoffSmPCA, thin_factor, window, plot_error=False)
        
        os.mkdir("CNN_vset_n{}".format(nperseg))
        Cache(vset_pa, vset_ta, vset_ka, vset_pataka, nperseg)
        Cache_llambda(llambda_samples)
    else:
        vset_pa, vset_ta, vset_ka, vset_pataka = OpenCache(nperseg)
        
    for [syll,vset] in [['pa', vset_pa]]:
        torch.cuda.empty_cache()
        
        """ Setup Model """
        model = Net()
        optimizer = Adam(model.parameters(), lr=0.04)
        criterion = CrossEntropyLoss()
        
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        
        """ Train Model """
        trainInput = TrainInput(vset_pa)
        trainInput.GetInputArrs(window, step, nperseg)
        trainInput.Decimate(fraction)
        trainInput.Split()
        train_x, val_x, train_y, val_y = trainInput.ConvertToTorch()
        
        n_epochs = 50
        for epoch in range(n_epochs):
            model = train(epoch, model, optimizer, criterion, train_x, val_x, train_y, val_y)
        
        """ Save Model """
        torch.save(model.state_dict(), "CNN_models/model_{}_n{}w{}".format(syll,nperseg,window))  