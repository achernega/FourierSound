import numpy as np
import matplotlib.pyplot as plt
import numpy.fft
import scipy.signal as sig
import scipy.stats as stat

y = np.loadtxt(fname = "whiteNoise_2000pt_0ms.txt")

#Fourier transform --------------------------------
Fs = 15500
L = len(y)
NFFT = int(2**np.ceil(np.log2(abs(L)))) #/ Next power of 2 from length of y
Y = numpy.fft.fft(y,n=NFFT)/L
f = Fs/2*np.linspace(0,1,NFFT/2+1)
ydb = 2*np.abs(Y[0:int(NFFT/2+1)])
ydb = -20*np.log10(ydb/max(ydb))

#Plots --------------------------------------------
plt.plot(y)
plt.show()
plt.plot(f,ydb)

#Peak Detection -----------------------------------
distBtwnPeaks = 15 #Min distance between peaks
minHeightLowestPeak = 65 #Height of lowest peak
peaks = sig.find_peaks(ydb, distance=distBtwnPeaks, height=minHeightLowestPeak)
tranPeaks = peaks[0]*max(f)/len(ydb)

#Sound Velocity Calculation -----------------------
usableThresh = 9 #Last best usable peak
usablePeaks = tranPeaks[:usableThresh]
usablePeaksXAxis = np.linspace(1,usableThresh,usableThresh)
regressionRes = stat.linregress(usablePeaksXAxis, usablePeaks)
slope = regressionRes[0]
soundVelocity = 2*(0.5588)*slope
print("Velocity of Sound in Air: ", soundVelocity)

plt.show() #/ Shows fourier db transform
