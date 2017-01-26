# Antipodal chirp keying performance in Laplace noise.
# H0: x[n] = A * sf0[n] + w[n]
# H1: x[n] = A * sf1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Laplace random variable with mean 0 and variance var.
# sf0[n] and sf1[n] are deterministic chirp signals.

from utils import *
import matplotlib.pyplot as plt

N = 1024  # number of data points.
M = 10000  # number of monte carlo trials.

enr_range = np.linspace(0, 16, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

# generate the deterministic signal.
Ts = 1 / 1000  # sampling period.
fs = 1 / Ts  # sampling frequency

t = np.arange(N) * Ts  # continuous time signal.

A = 1e-6  # small amplitude.
sf0n = linear_chirp(t[:int(N / 2)], 250, 0.5, 100)  # chirp signal with positive chirp rate.
sf0p = linear_chirp(t[int(N / 2):], 100, 1, 250)  # chirp signal with negative chirp rate.
sf0 = np.hstack((sf0n, sf0p))  # bi-orthogonal chirp signals. Signifies symbol `0`

sf1p = linear_chirp(t[:int(N / 2)], 100, 0.5, 250)  # chirp signal with positive chirp rate.
sf1n = linear_chirp(t[int(N / 2):], 250, 1, 100)  # chirp signal with negative chirp rate.
sf1 = np.hstack((sf1p, sf1n))  # bi-orthogonal chirp signals. Signifies symbol `1`
deltasf = sf1 - sf0

sp0 = linear_chirp(t, 100, 1, 250, phi=np.pi)  # chirp signal.
sp1 = -sp0
deltasp = sp1 - sp0

# numerically calculate probability of detection.
Pp = np.zeros_like(enr_range)
Pf = np.zeros_like(enr_range)
for k in range(d2.size):
    # variance corresponding to d2
    var = N * (A ** 2) / (2 * d2[k])

    # determine the threshold corresponding to gamma
    gamma = 0  # threshold for Bayesian detector

    # generate the datap.
    datap = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * sp0  # laplace noise
    dataf = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * sf0  # laplace noise

    # apply the detector.
    Tp = np.sqrt(2 / var) * np.sign(datap).dot(deltasp)  # NP detector.
    Tf = np.sqrt(2 / var) * np.sign(dataf).dot(deltasf)  # NP detector.

    Pp[k] = np.where(Tp > gamma)[0].size / M
    Pf[k] = np.where(Tf > gamma)[0].size / M

# analytically calculate probability of error.
Pep = Q(np.sqrt(2 * d2))
Pef = Q(np.sqrt(d2))

# plot the results.
fig1, ax = get_figure()
ax.plot(enr_range, Pp, '*', label=r'$acs \; monte \; carlo$')
ax.plot(enr_range, Pep, label=r'$acs \; analytic$')
ax.plot(enr_range, Pf, '*', label=r'$bcs \; monte \; carlo$')
ax.plot(enr_range, Pef, label=r'$bcs \; analytic$')
ax.set_xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$', fontsize=20)
ax.set_ylabel(r'$P_e$', fontsize=20)
ax.legend(loc='upper right')

# plot the results in logarithmic scale.
fig2, ax = get_figure()
ax.semilogy(enr_range, Pp, '*', label=r'$acs \; monte \; carlo$')
ax.semilogy(enr_range, Pep, label=r'$acs \; analytic$')
ax.semilogy(enr_range, Pf, '*', label=r'$bcs \; monte \; carlo$')
ax.semilogy(enr_range, Pef, label=r'$bcs \; analytic$')
ax.set_xlabel(r'$10\log_{10}\frac{NA^2}{2\sigma^2}$', fontsize=20)
ax.set_ylabel(r'$P_e$', fontsize=20)
ax.legend(loc='lower left')

# plt.title(r'$Binary \; Phase \; Shift \; Keying \; in \; WGN$')
plt.tight_layout()
plt.show()
