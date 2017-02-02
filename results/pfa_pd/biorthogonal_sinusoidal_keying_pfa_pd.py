# Biorthogonal chirp keying BER-ENR performance in Gaussian noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Gaussian random variable with zero mean.
# s0[n] and s1[n] are deterministic antipodal sinusoidal signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024  # number of data points.
M = 10000  # number of monte carlo trials.

pfa = np.logspace(-1, -7, 7)
enr_range = np.linspace(0, 20, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

for i in range(pfa.size):
    # generate the deterministic signal.

    Ts = 1 / 1000  # sampling period.
    fs = 1 / Ts  # sampling frequency

    A = 1.0  # amplitude.
    f0 = 100  # continuous frequency
    f1 = 250  # continuous frequency
    t = np.arange(N) * Ts  # continuous time signal.
    s0 = A * np.cos(2 * np.pi * f0 * t)  # deterministic signal.
    s1 = A * np.cos(2 * np.pi * f1 * t)  # deterministic signal.

    deltas = s1 - s0

    # numerically calculate probability of detection.
    P = np.zeros_like(enr_range)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * A ** 2 / (2 * d2[k])

        # determine the threshold corresponding to gamma
        gamma = Qinv(pfa[i]) * np.sqrt(var * N) - 0.5 * A * N

        # generate the data.
        data = np.random.normal(scale=np.sqrt(var), size=(M, N)) + A * s1

        # apply the detector.
        T = np.dot(data, deltas)
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i]) - np.sqrt(d2 * 2))

    # plot the results.
    plt.plot(enr_range, P, '*')
    plt.plot(enr_range, Pd, label='pfa = {}'.format(pfa[i]))

plt.xlabel(r'$10\log_{10}\frac{N A^2}{2\sigma^2}$')
plt.ylabel(r'$P_D$')
plt.grid()
plt.show()
