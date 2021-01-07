""" GetPeaksMatrix """

import scipy.signal as ss

def GetPeakVector(peaks, y, times):
    # this takes the input of ALL the peaks in an utterance
    
    prominences, lb, rb = ss.peak_prominences(y, peaks)
    
    widths02, wh, li, ri = ss.peak_widths(y, peaks, rel_height = 0.2)
    widths04, wh, li, ri = ss.peak_widths(y, peaks, rel_height = 0.4)
    widths05, wh, li, ri = ss.peak_widths(y, peaks, rel_height = 0.5)
    widths06, wh, li, ri = ss.peak_widths(y, peaks, rel_height = 0.6)
    widths08, wh, li, ri = ss.peak_widths(y, peaks, rel_height = 0.8)
    
    distances = [peaks[1]-peaks[0]]
    for i in range(1, len(peaks)-1):
        d = (0.5*(peaks[i]-peaks[i-1])**2 + 0.5*(peaks[i+1]-peaks[i])**2)**0.5
        distances.append(d)
    distances.append(peaks[-2] - peaks[-1])
    fps = len(times) / (times[-1] - times[0])
    distances = [distance/fps for distance in distances]
    
    growthLeft = [1.0]
    for i in range(1, len(peaks)):
        gl = y[peaks[i-1]]/y[peaks[i]]
        growthLeft.append(gl)
    
    growthRight = []
    for i in range(0, len(peaks)-1):
        gr = y[peaks[i+1]]/y[peaks[i]]
        growthRight.append(gr)
    growthRight.append(1.0)
    
    rate = len(peaks)/(times[-1] - times[0])
    
    peaksVectors = []
    for i, peak in enumerate(peaks):
        prominence = prominences[i]
        width02 = widths02[i]
        width04 = widths04[i]
        width05 = widths05[i]
        width06 = widths06[i]
        width08 = widths08[i]
        distance = distances[i]
        
        if (i > 0):
            leftDistance = distances[i-1]
        else:
            leftDistance = distances[i]
            
        if (i < len(peaks)-1):
            rightDistance = distances[i+1]
        else:
            rightDistance = distances[i]
            
        gl = growthLeft[i] - 1
        gr = growthRight[i] - 1
        rate = rate
        
        peakVector = [prominence, width02, width04, width05, width06, width08, distance, leftDistance, rightDistance, gl, gr, rate]
        peaksVectors.append(peakVector)
    # print(peaksVectors)
    return peaksVectors
    