
from utils import *

N = 10
M = 10000

pfa = np.logspace(-7, -1, 7)
enr = np.linspace(0, 20, 50)
d2 = 10**(enr/10)

for i in range(pfa.size):
    # generate the deterministic signal.
    A = 1  # amplitude.
    s = A*np.ones((M, N))  # deterministic dc level.

    # numerically calculate probability of detection.
    P = np.zeros_like(enr)
    for k in range(d2.size):
        # variance corresponding to d2
        var = N*A**2/d2[k]

        # generate the noise.
        w = np.sqrt(var)*np.random.randn(M, N)

        # generate the M-by-N data under H1 hypothesis.
        data = s + w   # make use of python broadcasting.

        # determine thresholds for M realizations.
        gamma = np.sqrt(var/N)*Qinv(pfa[i])*np.ones(M)  # should be M-by-1 vector.

        # apply the detector.
        T = data.mean(axis=1)  # should be M-by-1 vector.
        P[k] = np.where(T > gamma)[0].size / M

    # analytically calculate probability of detection.
    Pd = Q(Qinv(pfa[i]) - np.sqrt(d2))

    # plot the results.
    plt.plot(enr, P, '*')
    plt.plot(enr, Pd)


plt.grid()
plt.show()


