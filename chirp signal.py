# Time-frequency analysis of chirp signals.

from utils import *
from scipy import fftpack, signal
import matplotlib.pyplot as plt


N = 1024  # number of points.
Ts = 1 / 1000  # sampling period.
fs = 1 / Ts  # sampling frequency

t = np.arange(N)*Ts  # continuous time.
x = linear_chirp(t, 100, 5, 1250, phi=np.pi)  # chirp signal.
xf = fftpack.fft(x)  # fft of the signal
f = fftpack.fftfreq(N, Ts)  # get the sampled frequencies.

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(t, x)  # plot signal in time.
ax[0, 0].set_xlabel('$t \; [sec]$')
ax[0, 0].set_ylabel('$x(t)$')

ax[1, 0].plot(fftpack.fftshift(f), 1/N * np.abs(fftpack.fftshift(xf)))  # amplitude spectrum.
ax[1, 0].set_xlabel('$f \; [Hz]$')
ax[1, 0].set_ylabel('$|X(f)|$')
ax[1, 0].grid()

f, Pxx = signal.periodogram(x, fs)
ax[0, 1].semilogy(f, Pxx)
ax[0, 1].set_xlabel('$f \; [Hz]$')
ax[0, 1].set_ylabel('$PSD$')

ax[1, 1].specgram(x, NFFT=128, Fs=fs, noverlap=64)  # spectrogram of the signal.
ax[1, 1].set_xlabel('$t \; [sec]$')
ax[1, 1].set_ylabel('$f \; [Hz]$')

plt.tight_layout()
plt.show()