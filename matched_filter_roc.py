# Receiver Operating Characteristics (ROC) of
# detection of the signal s[n] = r**n, 0<r<1, in WGN.
# PD = Pr(T > gamma) = Q(Qinv(PFA) / sqrt(epsilon/var)))
# where
#   epsilon: the signal energy,
#   T: sample mean, i.e. T(x) = mean(x)
#   gamma: threshold.
#   Q: error function.
#   Qinv: inverse error function.
#   var: variance of the random variable.
#   N: number of data points.

from utils import *

np.random.seed(0)  # set seed of random number generator.

r = 0.9  # exponential decay rate.
N = 10  # the number of data points.
M = 10000  # number of realizations of the test statistic T.

s = np.zeros(N)
for n in range(N):
    s[n] = r**n  # exponential signal to be detected.


epsilon = s.dot(s)  # signal energy.
# S = np.array([s for i in range(M)]).T

for PFA in np.logspace(-7, -1, 7):
    enr = np.linspace(0, 20, 100)  # energy-to-noise ratio.
    d2 = 10**(enr/10)  # deflection coefficient of the detector.
    var = epsilon / d2
    gamma = np.sqrt(var*epsilon)*Qinv(PFA)  # threshold for a given gamma.

    P = np.zeros(enr.size)  # probability vector.
    for i in range(enr.size):
        data = np.sqrt(var[i])*np.random.randn(M, N) + s  # generate M-by-N random data.
        T = data.dot(s)  # compute the test statistic. Here it is sample mean.
        M_gamma = np.where(T > gamma[i])[0]  # number of T > gamma
        P[i] = M_gamma.size/M

    P_FA = Q(Qinv(PFA) - np.sqrt(d2))  # analytic value of Pr(T > gamma)

    plt.plot(enr, P, '*')
    plt.plot(enr, P_FA)

plt.xlabel(r'$10log_{10}\frac{\varepsilon}{\sigma^2}$')
plt.ylabel(r'$P_D = Pr\{T > \gamma\}$')
plt.title(r'$Pr\{T > \gamma\} = Q(Q^{-1}(P_{FA}) - \sqrt{\frac{\varepsilon}{\sigma^2}})$', y=1.04)
plt.grid()
plt.show()

