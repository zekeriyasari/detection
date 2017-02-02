# Antipodal chirp keying PFA-PD performance in Laplace noise.
# H0: x[n] = A * s0[n] + w[n]
# H1: x[n] = A * s1[n] + w[n],  n = 0, 1, ..., N-1
# w[N] is Laplace random variable with zero mean.
# s0[n] and s1[n] are deterministic antipodal chirp signals..

from utils import *
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.info('Started simulation...')

N = 1024  # number of data points.
M = 10000  # number of monte carlo trials.

pfa = np.linspace(0., 1., 50)
d2 = np.arange(0.25, 1.25, 0.25)

fig, ax = get_figure()

for i in range(len(d2)):

    logging.info('Started d2: {}'.format(d2[i]))

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
    Pp = np.zeros_like(pfa)
    Pf = np.zeros_like(pfa)
    for k in range(d2.size):

        logging.info('Started d2: {}'.format(d2[k]))

        # variance corresponding to d2
        var = N * A ** 2 / (2 * d2[i])

        # determine the threshold corresponding to gamma
        gamma = Qinv(pfa[k]) * np.sqrt(4 * N / var) - 2 * A * N / var

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
    Pp = Q(Qinv(pfa) - np.sqrt(d2[i] * 8))
    Pf = Q(Qinv(pfa) - np.sqrt(d2[i] * 4))

    # plot the results.
    ax.plot(pfa, Pp, '*')
    ax.plot(pfa, Pp, label=r'$ack-d^2={}$'.format(d2[i]))
    ax.plot(pfa, Pf, '*')
    ax.plot(pfa, Pf, label=r'$bck-d^2={}$'.format(d2[i]))

    logging.info('Finished d2: {}'.format(d2[i]))

ax.set_xlabel(r'$P_{FA}$', fontsize=20)
ax.set_ylabel(r'$P_D$', fontsize=20)

ax.legend(loc='lower right')
plt.tight_layout()

directory = os.getcwd() + '/figures'
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)
plt.savefig('roc_comparison')

logging.info('Finished simulation')
