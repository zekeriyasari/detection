# This module contains the utility functions used in the project.


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erfinv


def linear_chirp(t, f0, t1, f1, phi=0):
    """
    Generates the linear chirp signal `s(t) = cos(2 * pi * fc * t + mu * pi * t ** 2)`
    with the instantaneous frequency `fi(t) = fc + mu * t` where,
    mu = (f1 - f0)/t1
    t1 is the time when fi(t1) = f1

    :param t: array-like
        time array
    :param f0: float
        initial frequency
    :param t1: float
        the time when instantaneous frequency is when t = 0
    :param f1: float
        final frequency
    :param phi: float
        initial phase in radians.

    :return:ndarray,
        chirp signal s(t)
    """
    t = np.asarray(t)
    t0 = t[0]
    mu = (f1 - f0) / (t1 - t0)  # chirp rate.
    phase = 2 * np.pi * (f0 * t + 0.5 * mu * (t - t0) ** 2)  # instantaneous frequency
    phi *= np.pi / 180
    return np.cos(phase + phi)


def Q(x):
    """
    This function computes the right tail probability for a
     N(0, 1) random variable.

    :param x: array
        Real vector of values of random variable.
    :return: array
        Right tail probabilities.
    """

    return 0.5 * erfc(x / np.sqrt(2))


def Qinv(x):
    """
    This function computes the inverse Q function.

    :param x: array
        Real vector of right-tail probabilities.
    :return: array
        Real vector of values of random variable.
    """

    return np.sqrt(2) * erfinv(1 - 2 * x)


def get_figure():
    # construct the figure.
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, 21, 5))
    ax.set_xticks(np.arange(0, 21, 1), minor=True)
    ax.set_yticks(np.arange(0, 21, 0.25))
    ax.set_yticks(np.arange(0, 21, 0.0625), minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.75)
    return fig, ax
