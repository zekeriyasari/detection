# Known DC level detection performance in WGN
# H0: x[n] = w[n]
# H1: x[n] = A + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# A is deterministic and assumed to be positive.

from utils import *
import matplotlib.pyplot as plt

N = 10
M = 10000

pfa = np.linspace(0., 1., 50)
d2 = np.linspace(1.0, 10., 5)


for i in range(len(d2)):
    # generate the deterministic signal.
    A = 1  # amplitude.
    s = A * np.ones(N)  # deterministic dc level.

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa) - np.sqrt(d2[i]))

    # numerically calculate probability of detection.
    P = np.zeros_like(pfa)
    for k in range(len(pfa)):
        # variance corresponding to d2
        var = N * A ** 2 / d2[i]

        # determine the threshold corresponding to gamma
        gamma = np.sqrt(var / N) * Qinv(pfa[k])

        # generate the datap.
        data = np.sqrt(var) * np.random.randn(M, N) + s

        # apply the detector.
        T = data.mean(axis=1)  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # plot the results.
    plt.plot(pfa, Pd)
    plt.plot(pfa, P, '*')

plt.show()

