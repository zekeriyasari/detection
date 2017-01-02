# Matched filter performance in WGN
# H0: x[n] = w[n]
# H1: x[n] = s[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# s[n] is deterministic.

from utils import *
import matplotlib.pyplot as plt

N = 1024
M = 1000

pfa = np.logspace(-1, -1, 1)
enr_range = np.linspace(0, 20, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

for i in range(pfa.size):
    # generate the deterministic signal.

    Ts = 1 / 1000  # sampling period.
    fs = 1 / Ts  # sampling frequency

    t = np.arange(N) * Ts  # continuous time signal.

    A = 1e-6  # small signal amplitude.
    s = np.ones(t.shape)

    # numerically calculate probability of detection.
    P = np.zeros_like(enr_range)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * A ** 2 / d2[k]

        # determine the threshold corresponding to gamma
        gamma = Qinv(pfa[i]) * np.sqrt(2 * N / var)

        # generate the data.
        data = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * s

        # apply the detector.
        T = np.sqrt(2 / var) * np.sum(np.sign(data), axis=1)  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i]) - np.sqrt(d2 * 2))

    # plot the results.
    plt.plot(enr_range, P, '*')
    plt.plot(enr_range, Pd)

plt.xlabel(r'$10\log_{10}\frac{\varepsilon}{\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.title(r'$Damped \; Exponential \; in \; WGN$')
plt.grid()
plt.show()
