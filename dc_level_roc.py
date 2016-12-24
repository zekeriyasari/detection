# Receiver Operating Characteristics (ROC) of
# detection of DC level in WGN.
# PD = Pr(T > gamma) = Q(Qinv((gamma - A) / sqrt(var/N)))
# where
# T: sample mean, i.e. T(x) = mean(x)
# gamma: threshold.
# Q: error function.
# Qinv: inverse error function.
# A: mean value of random variable.
# var: variance of the random variable.
# N: number of data points.

from utils import *

np.random.seed(0)  # set seed of random number generator.

A = 1
var = 10  # variance of the random variable.
N = 10  # the number of data points.
M = 10000  # number of realizations of the test statistic T.
for PFA in np.logspace(-7, -1, 7):
    enr = np.linspace(0, 20, 100)
    d2 = 10**(enr/10)
    var = N*(A**2)/d2
    gamma = np.sqrt(var/N)*Qinv(PFA)

    P = np.zeros(gamma.size)
    for i in range(gamma.size):
        data = np.sqrt(var[i])*np.random.randn(M, N) + A  # generate M-by-N random data.
        T = data.mean(axis=1)  # compute the test statistic. Here it is sample mean.
        M_gamma = np.where(T > gamma[i])[0]  # number of T > gamma
        P[i] = M_gamma.size/M

    P_true = Q(Qinv(PFA) - np.sqrt(d2))  # analytic value of Pr(T > gamma)

    plt.plot(10*np.log10(d2), P, '--', label='$P_{monte carlo}$')
    plt.plot(10*np.log10(d2), P_true, label='$P_{true}$')

plt.xlabel(r'$10log_{10}\frac{NA^2}{\sigma^2}$')
plt.ylabel(r'$P_D = Pr\{T > \gamma\}$')
plt.title('Detection performance for DC level in WGN. Dashed curves are Monte Carlo simulation results.')
plt.show()

