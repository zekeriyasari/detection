from utils import *

N = 10
M = 10000

pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 20, 50)
d2 = 10 ** (enr / 10)

for i in range(pfa.size):
    # generate the deterministic signal.
    f = 1e3
    fs = 10e3
    F = f / fs
    A = np.random.randn(M, 1)  # random amplitude.
    phi = np.random.rand(M, 1) * np.pi  # random phase
    s0 = np.cos(2 * np.pi * F * np.arange(N)) * np.cos(phi) - np.sin(2 * np.pi * F * np.arange(N)) * np.sin(phi)
    s = A * s0
    ksi0 = np.cos(2 * np.pi * F * np.arange(N)) * np.ones((M, 1))
    ksi1 = np.sin(2 * np.pi * F * np.arange(N)) * np.ones((M, 1))

    # numerically calculate probability of detection.
    P = np.zeros_like(enr)
    for k in range(d2.size):
        # variance corresponding to d2
        epsilon0 = np.sum(s0 ** 2, axis=1).reshape((M, 1))
        epsilon = (A ** 2) * epsilon0
        var = epsilon / d2[k]

        # generate the noise.
        w = np.sqrt(var) * np.random.randn(M, N)

        # generate the M-by-N data under H1 hypothesis.
        data = s + w  # make use of python broadcasting.

        # determine thresholds for M realizations.
        gamma = var * np.log(1 / pfa[i])  # should be M-by-1 vector.

        # apply the detector.
        T = 1 / N * (np.sum(data * ksi0, axis=1).reshape((M, 1))) ** 2 + 1 / N * (np.sum(data * ksi1, axis=1).reshape(
            (M, 1))) ** 2  # should be M-by-1 vector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    # Pd = Q(Qinv(pfa[i] / 2) - np.sqrt(d2)) + Q(Qinv(pfa[i] / 2) + np.sqrt(d2))

    # plot the results.
    plt.plot(enr, P, '*')
    # plt.plot(enr, Pd)

plt.grid()
plt.show()
