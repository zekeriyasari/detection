<<<<<<< HEAD

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


A = 1
f = 1e3
fs = 10e3
T = 1 / f
Ts = 1 / fs
t = np.arange(0, T, Ts)
x = A * np.cos(2 * np.pi * f * t)

plt.plot(t, x)
plt.show()

=======
# Time-frequency analysis of chirp signals.

from utils import *
from scipy import fftpack, signal
import matplotlib.pyplot as plt

N = 1024  # number of points.
Ts = 1 / 1000  # sampling period.
fs = 1 / Ts  # sampling frequency

t = np.arange(N) * Ts  # continuous time.
xn = linear_chirp(t[:N/2], 250, 0.5, 100)  # chirp signal with positive chirp rate.
xp = linear_chirp(t[N/2:], 100, 1, 250)  # chirp signal with negative chirp rate.
x = np.hstack((xn, xp))  # bi-orthogonal chirp signals. Signifies symbol `0`

# xp = linear_chirp(t[:N/2], 100, 0.5, 250)  # chirp signal with positive chirp rate.
# xn = linear_chirp(t[N/2:], 250, 1, 100)  # chirp signal with negative chirp rate.
# x = np.hstack((xp, xn))  # bi-orthogonal chirp signals. Signifies symbol `1`

xf = fftpack.fft(x)  # fft of the signal
f = fftpack.fftfreq(N, Ts)  # get the sampled frequencies.

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(t, x)  # plot signal in time.
ax[0, 0].set_xlabel('$t \; [sec]$')
ax[0, 0].set_ylabel('$x(t)$')

ax[1, 0].plot(fftpack.fftshift(f), 1 / N * np.abs(fftpack.fftshift(xf)))  # amplitude spectrum.
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
>>>>>>> 7171669baf34094657360f669b3ce3f49329149e
