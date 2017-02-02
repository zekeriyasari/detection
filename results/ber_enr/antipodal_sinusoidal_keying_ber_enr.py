# Antipodal sinusoidal keying BER-ENR performance in Gaussian noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Gaussian random variable with zero mean.
# s0[n] and s1[n] are deterministic antipodal sinusoidal signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024  # number of points
M = 10000  # number of monte carlo trials

enr_range = np.linspace(0, 16, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

Ts = 1 / 1000  # sampling period
fs = 1 / Ts  # sampling frequency
t = np.arange(N) * Ts  # continuous time.
A = 1.0  # amplitude.
f = 100  # continuous time frequency
s0 = A * np.cos(2 * np.pi * f * t + np.pi)  # deterministic dc level.
s1 = -s0
deltas = s1 - s0

# numerically calculate probability of detection.
P = np.zeros_like(enr_range)
for k in range(d2.size):
    # variance corresponding to d2
    var = N * A ** 2 / (2 * d2[k])

    # determine the threshold corresponding to gamma
    gamma = 0  # threshold for Bayesian detector

    # generate the data.
    data = np.sqrt(var) * np.random.randn(M, N) + A * s0

    # apply the detector.
    T = data.dot(deltas)  # NP detector.
    P[k] = np.where(T > gamma)[0].size / M

# analytically calculate probability of error.
Pe = Q(np.sqrt(d2))

# plot the results.
plt.plot(enr_range, P, '*', label=r'$ask \; monte \; carlo$')
plt.plot(enr_range, Pe, label=r'$ask \; analytic$')
plt.xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$')
plt.ylabel(r'$P_e$')
plt.legend(loc='upper right')
plt.grid()

# plot the results in logarithmic scale.
plt.figure()
plt.semilogy(enr_range, P, '*', label=r'$ask \; monte carlo$')
plt.semilogy(enr_range, Pe, label=r'$ask \; analytic$')
plt.xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$')
plt.ylabel(r'$P_e$')
plt.grid()

plt.show()
