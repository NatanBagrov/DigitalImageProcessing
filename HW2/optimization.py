import numpy as np


def conjugate_transpose(array):
    return np.transpose(np.conjugate(array))


def alternating_direction_method_of_multipliers(A, f_step, l=1.0, p=1.0, epsilon=1e-2, callback=lambda _:False):
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

    #

    change = np.inf
    f = np.random.rand(A.shape[1])
    z = A @ f
    a = np.ones_like(z)

    while change > epsilon:
        previous_f = f
        f = f_step(a, z, p)
        z = shrink(A @ f + a / p, l / p)
        a += p * (A @ f - z)
        change = np.linalg.norm(f - previous_f) / np.maximum(np.linalg.norm(previous_f), 1.0)

        if callback(f):
            break

    return f


def de_degradation_f_step(y, K, A, p=1.0,):
    """

    \arg\min_{f} ||y - K @ f||_2^2 + \lambda || A f ||_1

    :param K: degradation matrix
    :param A: A is Psi^* - analysis operator
    :return:
    """

    A_star = conjugate_transpose(A)
    multiplier = np.linalg.inv(K @ conjugate_transpose(K) + p * A_star @ A)

    def f_step(a, z, rho):
        assert rho == p
        f = multiplier @ (y - A_star @ (a - z))

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
