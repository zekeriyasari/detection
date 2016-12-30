# Matched filter performance in WGN
# H0: x[n] = w[n]
# H1: x[n] = A*s[n, phi] + w[n],  n = 0, 1, ..., N-1
# w[N] is WGN with mean 0 and variance var.
# s[n] is deterministic and known.
# A is deterministic and unknown.
# phi is deterministic and unknown.


from utils import *
import matplotlib.pyplot as plt

N = 10
M = 10000

pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 20, 50)
d2 = 10 ** (enr / 10)

for i in range(pfa.size):
    # generate the deterministic signal.
    A = np.random.randn()  # unknown amplitude.
    phi = np.random.rand() * np.pi  # unknown phase.
    F = 0.25  # discrete frequency
    n = np.arange(N)
    s = A * np.cos(2 * np.pi * F * n + phi)

    ksi0 = np.cos(2 * np.pi * F * n)
    ksi1 = np.sin(2 * np.pi * F * n)

    # numerically calculate probability of detection.
    P = np.zeros_like(enr)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * (A ** 2) / (2 * d2[k])

        # determine the threshold corresponding to gamma
        gamma = var * np.log(1 / pfa[i])

        # generate the datap.
        data = np.sqrt(var) * np.random.randn(M, N) + s

        # apply the detector.
        T = 1 / N * (data.dot(ksi0) ** 2) + 1 / N * (data.dot(ksi1) ** 2)  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    # Pd = Q(Qinv(pfa[i] / 2) - np.sqrt(d2)) + Q(Qinv(pfa[i] / 2) + np.sqrt(d2))

    # plot the results.
    plt.plot(enr, P, '*')
    # plt.plot(enr, Pd)

plt.xlabel(r'$10\log_{10}\frac{N A^2}{2\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.title(r'$Unknown \; Amplitude \; and \; Phase \; Sinusoid \; in \; WGN$')
plt.grid()
plt.show()
