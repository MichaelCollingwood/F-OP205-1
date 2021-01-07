from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

def PlotSpecAndFunc(frequencies, times, Sxx, times_,  y, locations, manuallyScoredData):
    
    fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
    ax1.pcolormesh(times, frequencies, Sxx, shading='nearest', norm=LogNorm())
    
    ax2.plot(times_, y, color="black")
    for location in locations:
        ax2.scatter(times_[int(location)], y[int(location)], c="red", marker="x")    

    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.xlim(0,10)
    plt.title("{}: # real: {}, estimated: {}".format(manuallyScoredData[0], manuallyScoredData[3], len(locations)))
    plt.show()