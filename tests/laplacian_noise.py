import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-10, 10, 0.01)

mu = 0
for lamda in range(1, 5):
    p = 1 / (2 * lamda) * np.exp(-1 / lamda * np.abs(x - mu))
    plt.plot(x, p, label=r'$\mu = {}, \; \lambda = {}$'.format(mu, lamda))

mu = -5
lamda = 5
p = 1 / (2 * lamda) * np.exp(-1 / lamda * np.abs(x - mu))
plt.plot(x, p, label=r'$\mu = {}, \; \lambda = {}$'.format(mu, lamda))

plt.xlabel(r'$x$')
plt.ylabel(r'$p(x)$')
plt.legend()
plt.show()

