# Matched filter performance in WGN
# H0: x[n] = w[n]
# H1: x[n] = A*s[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# s[n] is deterministic and known.
# A is deterministic and unknown.

from utils import *

N = 10
M = 10000

pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 20, 50)
d2 = 10 ** (enr / 10)

for i in range(pfa.size):
    # generate the deterministic signal.
    A = np.random.randn()  # random amplitude.
    r = 0.5  # exponential decay rate.
    # s = A*np.array([r**n for n in range(N)])  # deterministic exponential signal.
    n = np.arange(N)
    F = 0.5
    s = A * np.cos(2 * np.pi * F * n)
    # s = A*np.array([r**n for n in range(N)])  # deterministic exponential signal.
    epsilon = s.dot(s)  # signal energy.

    # numerically calculate probability of detection.
    P = np.zeros_like(enr)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * (A ** 2) / d2[k]

        # determine the threshold corresponding to gamma
        gamma = var * (epsilon / A ** 2) * (Qinv(pfa[i] / 2) ** 2)

        # generate the data.
        data = np.sqrt(var) * np.random.randn(M, N) + s

        # apply the detector.
        T = data.dot(s / A) ** 2  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i] / 2) - np.sqrt(d2)) + Q(Qinv(pfa[i] / 2) + np.sqrt(d2))

    # plot the results.
    plt.plot(enr, P, '*')
    plt.plot(enr, Pd)

plt.xlabel(r'$10\log_{10}\frac{\varepsilon}{\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.title(r'$Unknown \; Amplitude \; Damped \; Exponential \; in \; WGN$')
plt.grid()
plt.show()
