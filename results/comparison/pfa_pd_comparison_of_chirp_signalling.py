# Antipodal chirp keying PFA-PD performance in Laplace noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Laplace random variable with zero mean.
# s0[n] and s1[n] are deterministic antipodal chirp signals..

from utils import *
import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.info('Started simulation...')

N = 1024  # number of data points.
M = 1000  # number of monte carlo trials.

pfa = np.logspace(-1, -7, 3)
enr_range = np.linspace(0, 20, 50)
d2 = np.array([10 ** (enr / 10) for enr in enr_range])

fig, ax = get_figure()

for i in range(pfa.size):

    logging.info('Started pfa: {}'.format(pfa[i]))

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

        logging.info('Started d2: {}'.format(d2[k]))

        # variance corresponding to d2
        var = N * A ** 2 / (2 * d2[k])

        # determine the threshold corresponding to gamma
        gamma = Qinv(pfa[i]) * np.sqrt(4 * N / var) - 2 * A * N / var

        # generate the data.
        datap = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * sp1  # laplace noise
        dataf = np.random.laplace(scale=np.sqrt(var / 2), size=(M, N)) + A * sf1  # laplace noise

        # apply the detector.
        Tp = np.sqrt(2 / var) * np.sign(datap).dot(deltasp)  # NP detector.
        Tf = np.sqrt(2 / var) * np.sign(dataf).dot(deltasf)  # NP detector.

        Pp[k] = np.where(Tp > gamma)[0].size / M
        Pf[k] = np.where(Tf > gamma)[0].size / M

        logging.info('Ended d2: {}'.format(d2[k]))

    # analytically calculate probability of detection.
    Pp = Q(Qinv(pfa[i]) - np.sqrt(d2 * 8))
    Pf = Q(Qinv(pfa[i]) - np.sqrt(d2 * 4))

    # plot the results.
    ax.plot(enr_range, Pp, '*')
    ax.plot(enr_range, Pp, label=r'$ack-pfa={}$'.format(pfa[i]))
    ax.plot(enr_range, Pf, '*')
    ax.plot(enr_range, Pf, label=r'$bck-pfa={}$'.format(pfa[i]))

    logging.info('Finished pfa: {}'.format(pfa[i]))

ax.set_xlabel(r'$10\log_{10}\frac{N A^2}{2\sigma^2}$', fontsize=20)
ax.set_ylabel(r'$P_D$', fontsize=20)

ax.legend(loc='lower right')
plt.tight_layout()

directory = os.getcwd() + '/figures'
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)
plt.savefig('pfa_pd_comparison')

logging.info('Finished simulation')
