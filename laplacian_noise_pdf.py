# Laplacian noise pdf
# p(x) = 1 / (2 lamda) exp(-1 / lamda abs(x - mu))
# where,
# lamda: is the exponential rate of decay.
# mu: location of the distribution peak

import numpy as np
import matplotlib.pyplot as plt

mu = 0  # location of the peak
x = np.arange(-10, 10, 0.01)
for lamda in range(1, 5):
    p = 1 / (2 * lamda) * np.exp(- 1 / lamda * np.abs(x - mu))
    plt.plot(x, p, label=r'$\mu = {},  \lambda = {}$'.format(mu, lamda))

mu = -5
lamda = 5
p = 1 / (2 * lamda) * np.exp(- 1 / lamda * np.abs(x - mu))
plt.plot(x, p, label=r'$\mu = {},  \lambda = {}$'.format(mu, lamda))

mu = 5
lamda = 5
p = 1 / (2 * lamda) * np.exp(- 1 / lamda * np.abs(x - mu))
plt.plot(x, p, label=r'$\mu = {},  \lambda = {}$'.format(mu, lamda))

plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$p(x)$')
plt.show()
