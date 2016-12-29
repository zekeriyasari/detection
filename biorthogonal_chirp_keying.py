# Antipodal chirp keying performance in Laplace noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Laplace random variable with mean 0 and variance var.
# s0[n] and s1[n] are deterministic chirp signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024
M = 10000

enr = np.linspace(0, 16, 50)
d2 = 10 ** (enr / 10)

# generate the deterministic signal.
Ts = 1 / 1000  # sampling period.
fs = 1 / Ts  # sampling frequency

t = np.arange(N)*Ts  # continuous time signal.

A = 1  # amplitude.
s0n = linear_chirp(t[:N/2], 250, 0.5, 100)  # chirp signal with positive chirp rate.
s0p = linear_chirp(t[N/2:], 100, 1, 250)  # chirp signal with negative chirp rate.
s0 = np.hstack((s0n, s0p))  # bi-orthogonal chirp signals. Signifies symbol `0`

s1p = linear_chirp(t[:N/2], 100, 0.5, 250)  # chirp signal with positive chirp rate.
s1n = linear_chirp(t[N/2:], 250, 1, 100)  # chirp signal with negative chirp rate.
s1 = np.hstack((s1p, s1n))  # bi-orthogonal chirp signals. Signifies symbol `1`
deltas = s1 - s0

epsilon_s0 = s0.dot(s0)
epsilon_s1 = s1.dot(s1)
epsilon_deltas = deltas.dot(deltas)


# numerically calculate probability of detection.
P = np.zeros_like(enr)
for k in range(d2.size):
    # variance corresponding to d2
    var = N * (A ** 2) / (2 * d2[k])

    # determine the threshold corresponding to gamma
    # gamma = np.sqrt(var/N) * Qinv(pfa[i])
    gamma = 0  # threshold for Bayesian detector

    # generate the data.
    # data = np.sqrt(var) * np.random.randn(M, N) + s0  # gaussian noise
    data = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + s0  # laplace noise

    # apply the detector.
    T = np.sign(data).dot(deltas)  # NP detector.
    P[k] = np.where(T > gamma)[0].size / M

# analytically calculate probability of error.
Pe = Q(np.sqrt(d2))

# plot the results.
plt.plot(enr, P, '*')
plt.plot(enr, Pe)
plt.xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$')
plt.ylabel(r'$P_e$')
plt.grid(True)

# plot the results in logarithmic scale.
plt.figure()
plt.semilogy(enr, P, '*')
plt.semilogy(enr, Pe)
plt.xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$')
plt.ylabel(r'$P_e$')
plt.grid(True)

# plt.title(r'$Binary \; Phase \; Shift \; Keying \; in \; WGN$')
plt.show()
