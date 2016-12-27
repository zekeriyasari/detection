# Known DC level detection performance in WGN
# H0: x[n] = w[n]
# H1: x[n] = A + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# A is deterministic, unknown, and assumed to be positive.

from utils import *

N = 10
M = 10000

pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 20, 50)
d2 = 10 ** (enr / 10)

for i in range(pfa.size):
    # generate the deterministic signal.
    A = np.random.randn()  # random amplitude.
    s = A * np.ones(N)  # deterministic dc level.

    # numerically calculate probability of detection.
    P = np.zeros_like(enr)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * A ** 2 / d2[k]

        # determine the threshold corresponding to gamma
        gamma = np.sqrt(var / N) * Qinv(pfa[i] / 2)

        # generate the data.
        data = np.sqrt(var) * np.random.randn(M, N) + s

        # apply the detector.
        T = np.abs(data.mean(axis=1))  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i] / 2) - np.sqrt(d2)) + Q(Qinv(pfa[i] / 2) + np.sqrt(d2))

    # plot the results.
    plt.plot(enr, P, '*')
    plt.plot(enr, Pd)

plt.xlabel(r'$10\log_{10}\frac{NA^2}{\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.title(r'$Unknown \; DC \; Level \; in \; WGN$')
plt.grid()
plt.show()
