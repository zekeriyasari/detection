# Phase sh
# H0: x[n] = A*s0[n] + w[n]
# H1: x[n] = A*s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# s0[n] and s1[n] is deterministic.

from utils import *
import matplotlib.pyplot as plt

N = 1024
M = 10000

pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 20, 50)
enr_range = np.linspace(0, 16, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

for i in range(pfa.size):
    # generate the deterministic signal.

    Ts = 1 / 1000  # sampling period.
    fs = 1 / Ts  # sampling frequency

    t = np.arange(N) * Ts  # continuous time signal.

    A = 1  # amplitude.
    f = 100  # frequency
    s0 = np.cos(2 * np.pi * f * t + np.pi)  # chirp signal.
    s1 = -s0
    deltas = s1 - s0

    # numerically calculate probability of detection.
    P = np.zeros_like(enr)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * A ** 2 / (2 * d2[k])

        # determine the threshold corresponding to gamma
        # gamma = Qinv(pfa[i]) * np.sqrt(4 * N / var) - 2 * A * N / var
        gamma = Qinv(pfa[i]) * np.sqrt(2 * var * N) - A * N

        # generate the datap.
        data = np.random.normal(scale=np.sqrt(var), size=(M, N)) + A * s1

        # apply the detector.
        # T = np.sign(data).dot(deltas)  # NP detector.
        T = data.dot(deltas)  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i]) - np.sqrt(d2 * 4))

    # plot the results.
    plt.plot(enr, P, '*')
    plt.plot(enr, Pd)

plt.xlabel(r'$10\log_{10}\frac{\varepsilon}{\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.title(r'$Damped \; Exponential \; in \; WGN$')
plt.grid()
plt.show()
