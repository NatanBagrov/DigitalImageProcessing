from functools import reduce

import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.signal import correlate2d
from tqdm import tqdm

from utils import conjugate_transpose

# TODO: DEBUG THIS MODULE! CURRENTLY NOT IN USE


def alternating_direction_method_of_multipliers(A, f, f_step, l=1.0, p=1.0, epsilon=1e-2, callback=lambda _: False):
    """
    L_p(f, z, a)=g(f)+h(z)+a^T(Af-z)+p/2(Af+z)^T(Af+z)

    :param A: A is Psi^* - analysis operator
    :param f_step: closed form optimization of L_p by f
    :param l:
    :param p:
    :param epsilon:
    :param callback:

    :return:
    """

    change = np.inf
    # f = np.random.rand(*f.shape)
    z = A(f)
    a = [np.zeros_like(zz) for zz in z]
    bar = tqdm(desc='ADMM')

    while change > epsilon:
        previous_f = f
        f = f_step(a, z, p)
        a_f = A(f)
        z = [shrink(aa_ff + aa / p, l / p) for aa_ff, aa in zip(a_f, a)]
        a += [p * (a_f - zz) for zz in z]
        change = np.linalg.norm(f - previous_f) / np.maximum(np.linalg.norm(f), 1.0)
        bar.update(1)

        if callback(f):
            break

    bar.close()

    return f


def de_degradation_f_step_multiplication(y, K, A, p=1.0):
    """

    \arg\min_{f} ||y - K @ f||_2^2 + \lambda || A f ||_1

    :param K: degradation matrix
    :param A: A is Psi^* - analysis operator
    :return:
    """

    A_star = conjugate_transpose(A)
    K_star = conjugate_transpose(K)
    multiplier = np.linalg.inv(K_star @ K + p * A_star @ A)
    addendum = K_star @ y

    def f_step(a, z, rho):
        assert rho == p
        f = multiplier @ (addendum - A_star @ (a - z))

        return f

    return f_step


def de_degradation_f_step_convolution(y, K, A, p=1.0):
    k_frequency = fft2(K, shape=y.shape)
    a_frequency = list(map(lambda aa: fft2(aa, shape=y.shape), A))
    divider = np.square(np.abs(k_frequency)) / p + reduce(np.add, map(np.square, map(np.abs, a_frequency)))
    # divider = np.maximum(divider, 1e-1)
    # convolution with transposed kernel is correlation i think
    addendum = correlate2d(y, K, mode='same')

    def f_step(a, z, rho):
        assert rho == p
        # convolution with transposed kernel is correlation i think
        f = [correlate2d(aa - zz, AA, mode='same') for aa, zz, AA in zip(a, z, A)]
        f = reduce(np.add, f)
        f = addendum - f
        f = np.real(ifft2(fft2(f) / divider))

        return f

    return f_step


def shrink(x, l):
    """
    S_l(x)

    :param x:
    :param l:
    :return:
    """

    return np.require(x > l, int) * (x - l) + np.require(x < - l, int) * (x + l)
