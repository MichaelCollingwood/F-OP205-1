import numpy as np
import json

def Cache(vset_pa, vset_ta, vset_ka, vset_pataka, nperseg):
    """
    Cache Validated Sets
    """
    
    with open("CNN_vset_n{}/vset_pa".format(nperseg), 'w') as fp:
        json.dump(vset_pa, fp, indent=2)
        
    with open("CNN_vset_n{}/vset_ta".format(nperseg), 'w') as fp:
        json.dump(vset_ta, fp, indent=2)
        
    with open("CNN_vset_n{}/vset_ka".format(nperseg), 'w') as fp:
        json.dump(vset_ka, fp, indent=2)
        
    with open("CNN_vset_n{}/vset_pataka".format(nperseg), 'w') as fp:
        json.dump(vset_pataka, fp, indent=2)
        
def OpenCache(nperseg):
    """
    Open Cached Validated Sets
    """   
    
    with open("CNN_vset_n{}/vset_pa".format(nperseg), "r") as read_file:
        vset_pa = json.load(read_file)
            
    with open("CNN_vset_n{}/vset_ta".format(nperseg), "r") as read_file:
        vset_ta = json.load(read_file)
        
    with open("CNN_vset_n{}/vset_ka".format(nperseg), "r") as read_file:
        vset_ka = json.load(read_file)
        
    with open("CNN_vset_n{}/vset_pataka".format(nperseg), "r") as read_file:
        vset_pataka = json.load(read_file)
        
    return vset_pa, vset_ta, vset_ka, vset_pataka
    
def Cache_llambda(llambda_samples):
    with open("llambda_samples", "w") as fp:
        json.dump(llambda_samples, fp, indent=2)
        
def OpenCache_llambda():
    with open("llambda_samples", "r") as read_file:
        return np.array(json.load(read_file))