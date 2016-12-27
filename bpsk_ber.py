# BPSK performance in WGN.
# H0: x[n] = s0[n] + w[n]
# H1: x[n] = s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# s0[n] and s1[n] are deterministic.

from utils import *
import matplotlib.pyplot as plt

N = 10
M = 10000

# pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 16, 50)
d2 = 10 ** (enr / 10)

# for i in range(pfa.size):
# generate the deterministic signal.
A = 1  # amplitude.
F = 10  # discrete-frequency.
n = np.arange(N)
s0 = A * np.cos(2 * np.pi * F * n)  # deterministic dc level.
s1 = -s0
epsilon = s0.dot(s0)

# numerically calculate probability of detection.
P = np.zeros_like(enr)
for k in range(d2.size):
    # variance corresponding to d2
    var = epsilon / d2[k]

    # determine the threshold corresponding to gamma
    # gamma = np.sqrt(var/N) * Qinv(pfa[i])
    gamma = 0  # threshold for Bayesian detector

    # generate the data.
    data = np.sqrt(var) * np.random.randn(M, N) + s0

    # apply the detector.
    T = 2 * data.dot(s1)  # NP detector.
    P[k] = np.where(T > gamma)[0].size / M

# analytically calculate probability of error.
Pe = Q(np.sqrt(d2))

# plot the results.
plt.semilogy(enr, P, '*')
plt.semilogy(enr, Pe)

plt.xlabel(r'$10\log_{10}\frac{\varepsilon}{\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.title(r'$Binary \; Phase \; Shift \; Keying \; in \; WGN$')
plt.grid()
plt.show()
