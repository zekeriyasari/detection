# Probability of error (Pe) versus energy-to-noise(enr) performance of
# Binary Phase Shift Keying(BPSK)in WGN.
# PD = Pr(T > 0) = Q(sqrt(epsilon/var))
# where
#   epsilon: the signal energy,
#   T: sample mean, i.e. T(x) = mean(x)
#   Q: error function.
#   Qinv: inverse error function.
#   var: variance of the random variable.
#   N: number of data points.


from utils import *

N = 10  # number of data points.
M = 10000  # number of realizations of the data.

A = 1  # amplitude
f = 1e3  # frequency
fs = 10e3  # sampling frequency.
n = np.arange(N)
s0 = A*np.cos(2*np.pi*f/fs*n)  # signal under null null hypothesis
s1 = -A*np.cos(2*np.pi*f/fs*n)  # signal under alternative hypothesis

epsilon = s0.dot(s0)  # signal energy

enr = np.linspace(0, 12, 100)  # enr levels
d2 = 10**(enr/10)  # deflection coefficient of the detector.
var = epsilon / d2

P = np.zeros(enr.size)  # probability vector.
for i in range(enr.size):
    data = np.sqrt(var[i])*np.random.randn(M, N) + s0  # realization of the data under null hypothesis
    T = data.dot(s1 - s0)  # compute the test statistic.
    M_gamma = np.where(T > 0)[0]  # number of T > 0
    P[i] = M_gamma.size/M

Pe = Q(np.sqrt(epsilon/var))  # Probability of error

plt.semilogy(enr, P, '*')
plt.semilogy(enr, Pe)

plt.xlabel(r'$10log_{10}\frac{\varepsilon}{\sigma^2}$')
plt.ylabel(r'$P_e = Pr\{T > 0\}$')
plt.title(r'$Pr\{T > 0\} = Q(\sqrt{\varepsilon/\sigma^2})$', y=1.04)
plt.grid()
plt.show()

