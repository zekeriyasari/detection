import numpy as np
from scipy.special import erfc, erfinv
import matplotlib.pyplot as plt



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



