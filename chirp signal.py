# A simple FFT implementation.


import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

N = 1024  # number of points. Should be large enough for proper periodic sine implementation.
Ts = 1 / 1000  # sampling period.
fs = 1 / Ts  # sampling frequency

t = np.arange(N)*Ts  # continuous time.
x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 400 * t)  # signal
xf = fftpack.fft(x)  # fft of the signal
f = fftpack.fftfreq(N, Ts)  # get the sampled frequencies.


fig, ax = plt.subplots(2)
ax[0].plot(t, x)  # plot signal in time.
ax[0].set_xlabel('$t \; [sec]$')
ax[0].set_ylabel('$x(t)$')

ax[1].plot(fftpack.fftshift(f), 1/N * np.abs(fftpack.fftshift(xf)))  # amplitude spectrum.
ax[1].set_xlabel('$f \; [Hz]$')
ax[1].set_ylabel('$|X(f)|$')
ax[1].grid()

plt.tight_layout()
plt.show()