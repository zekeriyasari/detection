# Biorthogonal chirp keying PFA-PD performance in Laplace noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Laplace random variable with zero mean.
# s0[n] and s1[n] are deterministic biorthogonal chirp signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024  # number of data points
M = 10000  # number of monte carlo trials

pfa = np.logspace(-1, -7, 7)
enr_range = np.linspace(0, 20, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

for i in range(pfa.size):
    # generate the deterministic signal
    Ts = 1 / 1000  # sampling period
    fs = 1 / Ts  # sampling frequency

    t = np.arange(N) * Ts  # continuous time signal

    A = 1e-6  # small amplitude

    s0n = linear_chirp(t[:int(N / 2)], 250, 0.5, 100)  # chirp signal with positive chirp rate.
    s0p = linear_chirp(t[int(N / 2):], 100, 1, 250)  # chirp signal with negative chirp rate.
    s0 = np.hstack((s0n, s0p))  # bi-orthogonal chirp signals. Signifies symbol `0`

    s1p = linear_chirp(t[:int(N / 2)], 100, 0.5, 250)  # chirp signal with positive chirp rate.
    s1n = linear_chirp(t[int(N / 2):], 250, 1, 100)  # chirp signal with negative chirp rate.
    s1 = np.hstack((s1p, s1n))  # bi-orthogonal chirp signals. Signifies symbol `1`

    deltas = s1 - s0

    # numerically calculate probability of detection.
    P = np.zeros_like(enr_range)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N * A ** 2 / (2 * d2[k])

        # determine the threshold corresponding to gamma
        gamma = Qinv(pfa[i]) * np.sqrt(2 * N / var) - A * N / var

        # generate the data.
        data = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * s1

        # apply the detector.
        T = np.sqrt(2 / var) * np.sign(data).dot(deltas)  # NP detector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i]) - np.sqrt(d2 * 4))

    # plot the results.
    plt.plot(enr_range, P, '*')
    plt.plot(enr_range, Pd, label=r'$pfa={}$'.format(pfa[i]))

plt.xlabel(r'$10\log_{10}\frac{N A^2}{2\sigma^2}$')
plt.ylabel(r'$P_D$')

plt.legend(loc='lower right')
plt.grid()
plt.show()
