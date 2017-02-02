# Antipodal chirp keying BER-ENR performance in Laplace noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Laplace random variable with zero mean.
# s0[n] and s1[n] are deterministic antipodal chirp signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024  # number of data points.
M = 10000  # number of monte carlo trials.

enr_range = np.linspace(0, 16, 50)
d2 = np.array([10 ** (enr/10) for enr in enr_range])

# generate the deterministic signal.
Ts = 1 / 1000  # sampling period.
fs = 1 / Ts  # sampling frequency

t = np.arange(N)*Ts  # continuous time signal.

A = 1e-6  # small amplitude.
s0 = linear_chirp(t, 100, 1, 250, phi=np.pi)  # chirp signal.
s1 = -s0
deltas = s1 - s0


# numerically calculate probability of detection.
P = np.zeros_like(enr_range)
for k in range(d2.size):
    # variance corresponding to d2
    var = N * (A ** 2) / (2 * d2[k])

    # determine the threshold corresponding to gamma
    gamma = 0  # threshold for Bayesian detector

    # generate the datap.
    data = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * s0  # laplace noise

    # apply the detector.
    T = np.sqrt(2 / var) * np.sign(data).dot(deltas)  # NP detector.
    P[k] = np.where(T > gamma)[0].size / M

# analytically calculate probability of error.
Pe = Q(np.sqrt(2 * d2))

# plot the results.
plt.plot(enr_range, P, '*', label=r'$ack \; monte \; carlo$')
plt.plot(enr_range, Pe, label=r'$ack \; analytic$')
plt.xlabel(r'$10\log_{10}\frac{N A^2}{2\sigma^2}$')
plt.ylabel(r'$P_e$')
plt.legend(loc='upper right')
plt.grid()

# plot the results in logarithmic scale.
plt.figure()
plt.semilogy(enr_range, P, '*', label=r'$ack \; analytic$')
plt.semilogy(enr_range, Pe, label=r'$ack \; analytic$')
plt.xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$')
plt.ylabel(r'$P_e$')

plt.legend(loc='lower left')
plt.grid()
plt.show()
