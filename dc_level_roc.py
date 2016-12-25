# Receiver Operating Characteristics (ROC) of
# detection of DC level in WGN.
# PD = Pr(T > gamma) = Q(Qinv(Pfa) / sqrt(N A**2/var)))
# where
# T: sample mean, i.e. T(x) = mean(x)
# gamma: threshold.
# Q: error function.
# Qinv: inverse error function.
# A: dc level,
# var: variance of the random variable.
# N: number of data points.

from utils import *

np.random.seed(0)  # set seed of random number generator.

A = 10  # DC level to be detected.
N = 10  # the number of data points.
M = 10000  # number of realizations of the test statistic T.

for PFA in np.logspace(-7, -1, 7):
    enr = np.linspace(0, 20, 100)  # energy-to-noise ratio.
    d2 = 10**(enr/10)  # deflection coefficient of the detector.
    var = N*(A**2)/d2  # variance of the data.
    gamma = np.sqrt(var/N)*Qinv(PFA)  # threshold for a given gamma.

    P = np.zeros(gamma.size)  # probability vector.
    for i in range(gamma.size):
        data = np.sqrt(var[i])*np.random.randn(M, N) + A  # generate M-by-N random data.
        T = data.mean(axis=1)  # compute the test statistic. Here it is sample mean.
        M_gamma = np.where(T > gamma[i])[0]  # number of T > gamma
        P[i] = M_gamma.size/M

    P_true = Q(Qinv(PFA) - np.sqrt(d2))  # analytic value of Pr(T > gamma)

    plt.plot(enr, P, '*', label='$P_{monte carlo}$')
    plt.plot(enr, P_true, label='$P_{true}$')

plt.xlabel(r'$10log_{10}\frac{NA^2}{\sigma^2}$')
plt.ylabel(r'$P_D = Pr\{T > \gamma\}$')
plt.title(r'$Pr\{T > \gamma\} = Q(Q^{-1}(P_{FA}) - \sqrt{\frac{N A^2}{\sigma^2}} )$', y=1.04)
plt.grid()
plt.show()

