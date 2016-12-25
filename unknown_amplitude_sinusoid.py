# Receiver Operating Characteristics (ROC) of
# detection of the signal s[n] = A*cos(2*pi*f*t + phi) in WGN.
# with unknown amplitude.
# PD = Pr(T > gamma) = Q(Q^{-1}(P_{FA}/2 - \sqrt{\frac{\varepsilon}{\sigma^2}}) +
#                       Q(Q^{-1}(P_{FA}/2 + \sqrt{\frac{\varepsilon}{\sigma^2}})
# where
#   epsilon: the signal energy,
#   T: sample mean, i.e. T(x) = mean(x)
#   gamma: threshold.
#   Q: error function.
#   Qinv: inverse error function.
#   var: variance of the random variable.
#   N: number of data points.

from utils import *

# np.random.seed(0)  # set seed of random number generator.

N = 10  # the number of data points.
M = 10000  # number of realizations of the test statistic T.

A = np.random.randn()  # random unknown amplitude.
f = 1e3  # frequency
fs = 10e3  # sampling frequency.
n = np.arange(N)
s = A * np.cos(2 * np.pi * f / fs * n + np.pi)  # signal under null hypothesis

epsilon = s.dot(s)  # signal energy.

for PFA in np.logspace(-7, -1, 7):
    enr = np.linspace(0, 20, 100)  # energy-to-noise ratio.
    d2 = 10 ** (enr / 10)  # deflection coefficient of the detector.
    var = N * A ** 2 / (2*d2)
    gamma = var * (s / A).dot(s / A) * Qinv(PFA / 2) ** 2  # threshold for a given gamma.

    P = np.zeros(enr.size)  # probability vector.
    for i in range(enr.size):
        data = np.sqrt(var[i]) * np.random.randn(M, N) + s  # generate M-by-N random data.
        T = data.dot(s / A) ** 2  # compute the test statistic. Here it is sample mean.
        M_gamma = np.where(T > gamma[i])[0]  # number of T > gamma
        P[i] = M_gamma.size / M

    P_FA = Q(Qinv(PFA / 2) - np.sqrt(epsilon / var)) + Q(
        Qinv(PFA / 2) + np.sqrt(epsilon / var))  # analytic value of Pr(T > gamma)

    plt.plot(enr, P, '*')
    plt.plot(enr, P_FA)

plt.xlabel(r'$10log_{10}\frac{N A^2}{2\sigma^2}$')
plt.ylabel(r'$P_D = Pr\{T > \gamma\}$')
plt.title(
    r'$Pr\{T > \gamma\} = Q(Q^{-1}(P_{FA}/2 - \sqrt{\frac{\varepsilon}{\sigma^2}}) +'
    r' Q(Q^{-1}(P_{FA}/2 + \sqrt{\frac{\varepsilon}{\sigma^2}})$', y=1.04)
plt.grid()
plt.show()
