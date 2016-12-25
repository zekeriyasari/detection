# Receiver Operating Characteristics (ROC) of
# detection of the signal s[n] = A*cos(2*pi*f*t + phi) in WGN.
# with unknown amplitude and phase.
# PD = Pr(T > gamma) = Q_{\chi_2^2(\lambda)}(2\ln{\frac{1}{P_{FA}}})
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


A = np.random.randn()  # random amplitude N(0, 1) with normal distribution.
phi = np.random.rand()*np.pi  # random phase U(-pi, pi) with uniform distribution.
f = 1e3  # frequency
fs = 10e3  # sampling frequency.
F = f/fs
n = np.arange(N)
s = A * np.cos(2 * np.pi * f / fs * n + phi)
s0 = np.cos(2 * np.pi * F * n)
s1 = np.sin(2 * np.pi * F * n)

for Pfa in np.logspace(-7, -1, 7):
    enr = np.linspace(0, 20, 100)  # energy-to-noise ratio.
    d2 = 10 ** (enr / 10)  # deflection coefficient of the detector.
    var = N * A**2 / (2 * d2)
    gamma = var * np.log(1/Pfa)  # threshold for a given gamma.

    P = np.zeros(enr.size)  # probability vector.
    for i in range(enr.size):
        data = np.sqrt(var[i]) * np.random.randn(M, N) + s  # generate M-by-N random data.
        T = 1 / N * (data.dot(s0) ** 2 + data.dot(s1) ** 2)  # compute the test statistic. Here it is sample mean.
        M_gamma = np.where(T > gamma[i])[0]  # number of T > gamma
        P[i] = M_gamma.size / M

    plt.plot(enr, P, '*')

plt.xlabel(r'$10log_{10}\frac{N A^2}{2\sigma^2}$')
plt.ylabel(r'$P_D = Pr\{T > \gamma\}$')
plt.title(r'$Pr\{T > \gamma\} = Q_{\chi_2^2(\lambda)}(2\ln{\frac{1}{P_{FA}}})$', y=1.04)
plt.grid()
plt.show()
