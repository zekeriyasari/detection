# Antipodal sinusoidal keying PFA-PD performance in Gaussian noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Gaussian random variable with zero mean.
# s0[n] and s1[n] are deterministic antipodal sinusoidal signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024  # number of data points.
M = 10000  # number of monte carlo trials.

pfa = np.linspace(0., 1., 50)
d2 = np.arange(0.25, 1.25, 0.25)

for i in range(len(d2)):
    # generate the deterministic signal.
    Ts = 1 / 1000  # sampling period.
    fs = 1 / Ts  # sampling frequency

    t = np.arange(N) * Ts  # continuous time signal.

    A = 1e-6  # amplitude.
    f = 100  # frequency
    s0 = np.cos(2 * np.pi * f * t + np.pi)  # chirp signal.
    s1 = -s0
    deltas = s1 - s0

    # numerically calculate probability of detection.
    P = np.zeros_like(pfa)
    for k in range(len(pfa)):
        # variance corresponding to d2
        var = N * A ** 2 / (2 * d2[i])

        # determine the threshold corresponding to gamma
        gamma = Qinv(pfa[k]) * np.sqrt(2 * var * N) - A * N

        # generate the data.
        data = np.random.normal(scale=np.sqrt(var), size=(M, N)) + A * s1

        # apply the detector.
        T = np.dot(data, deltas)
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa) - np.sqrt(d2[i] * 4))

    # plot the results.
    plt.plot(pfa, P, '*')
    plt.plot(pfa, Pd, label='pfa = {}'.format(d2[i]))

plt.legend(loc='lower right')
plt.xlabel(r'$P_{FA}$')
plt.ylabel(r'$P_D$')
plt.grid()
plt.show()
