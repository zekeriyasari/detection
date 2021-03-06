# This program is a Monte Carlo simulation
# to compute Pr(T > gamma) = Q(gamma / sqrt(var/N))
# where,
# T: the test statistic,
# gamma: the threshold,
# Q: the error function,
# var: variance of the random variable,
# N: the number of datap points.

from utils import *
import matplotlib.pyplot as plt

np.random.seed(0)  # set seed of random number generator.

var = 10  # variance of the random variable.
N = 10  # the number of datap points.
M = 1000  # number of realizations of the test statistic T.

data = np.sqrt(var) * np.random.randn(M, N)  # generate M-by-N random datap.
T = data.mean(axis=1)  # compute the test statistic. Here it is sample mean.

n_gamma = 100  # number of thresholds to be tested.
gamma = np.linspace(T.min(), T.max(), n_gamma)  # threshold interval.

P = np.zeros(gamma.size)
for i in range(gamma.size):
    M_gamma = np.where(T > gamma[i])[0]  # number of T > gamma
    P[i] = M_gamma.size / M

P_true = Q(gamma / np.sqrt(var / N))  # analytic value of Pr(T > gamma)

plt.plot(gamma, P, '*', gamma, P_true)
plt.xlabel('$\gamma$')
plt.ylabel('$Pr\{T > \gamma\}$')
plt.title(r'$Pr\{T > \gamma\} = Q(\frac{\gamma}{\sqrt{\sigma^2/N}})$', y=1.04)
plt.grid()
plt.show()
