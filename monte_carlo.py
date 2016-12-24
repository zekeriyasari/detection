# This program is a Monte Carlo simulation
# to compute Pr(T > gamma) = Q(gamma / sqrt(var/N))
# where,
# T: the test statistic,
# gamma: the threshold,
# Q: the error function,
# var: variance of the random variable,
# N: the number of data points.


import numpy as np
from scipy.special import erfc, erfinv
import matplotlib.pyplot as plt


def Q(x):
    """
    This function computes the right tail probability for a
     N(0, 1) random variable.

    :param x: array
        Real vector of values of random variable.
    :return: array
        Right tail probabilities.
    """

    return 0.5*erfc(x/np.sqrt(2))


def Qinv(x):
    """
    This function computes the inverse Q function.

    :param x: array
        Real vector of right-tail probabilities.
    :return: array
        Real vector of values of random variable.
    """

    return np.sqrt(2)*erfinv(1 - 2*x)

np.random.seed(0)  # set seed of random number generator.

var = 10  # variance of the random variable.
N = 10  # the number of data points.
M = 1000  # number of realizations of the test statistic T.

data = np.sqrt(var)*np.random.randn(M, N)  # generate M-by-N random data.

T = data.mean(axis=1)  # compute the test statistic. Here it is sample mean.

n_gamma = 1000  # number of thresholds to be tested.
gamma = np.linspace(T.min(), T.max(), n_gamma)  # threshold interval.

P = np.zeros(T.size)
for i in range(n_gamma):
    M_gamma = np.where(T > gamma[i])[0]
    P[i] = M_gamma.size/M

P_true = Q(gamma/np.sqrt(var/N))

plt.plot(gamma, P, label='$P$')
plt.plot(gamma, P_true, label='$P_{\text{true}}$')
plt.xlabel('$\gamma$')
plt.ylabel('$Pr\{T > \gamma\}$')
plt.legend()
plt.show()









