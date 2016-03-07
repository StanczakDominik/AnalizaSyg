import numpy as np
import scipy.io.wavfile as wf
import scipy.fftpack as fft
import matplotlib.pyplot as plt
rate, data = wf.read("chord.wav")
data = data[84400:]
n = len(data)
freqbase = rate * ((np.arange(n) + n//2)%n - n//2)/n
spectrum = fft.fft(data)
spectrum = spectrum * spectrum.conjugate()
bin1 = np.min(np.where(freqbase>16))
bin2 = np.min(np.where(freqbase>4000))-1
spectrum = spectrum[bin1:bin2]
freqbase = freqbase[bin1:bin2]

peakbin = np.argmax(spectrum)
fPeak = freqbase[peakbin]
fActual_Hz = 440*2**(-9/12)

print(rate, data, n, freqbase, spectrum, fPeak, fActual_Hz, sep="\n")
plt.title("wat")
plt.plot(np.log10(freqbase), 10*np.log10(spectrum+1e-15))
plt.scatter(np.log10(fPeak), 10*np.log10(spectrum[peakbin]+1e-15))
plt.xlabel("f/Hz [log]")
plt.ylabel("p/dB")
plt.show()
