# This module contains the utility functions used in the project.


import numpy as np
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
    phi *= np. pi / 180
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


def linear_chirp(t, f0, t1, f1, phi=0):
    """Frequency-swept cosine generator.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    phi : float, optional
        Phase offset, in degrees. Default is 0.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.
        ``f(t) = f0 + (f1 - f0) * t / t1``
    """

    t = np.asarray(t)
    mu = (f1 - f0) / t1  # chirp rate
    phase = 2 * np.pi * (f0 * t + 0.5 * mu * t * t)  # instantaneous phase.
    phi *= np.pi / 180  # Convert  phi to radians.
    return np.cos(phase + phi)



