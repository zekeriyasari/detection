
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

