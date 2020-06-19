import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    # TODO: implement.
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        return 0.5 * Ax_b.dot(Ax_b)

    def grad(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        return self.matvec_ATx(Ax_b)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    # TODO: implement.
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        alpha *= self.regcoef
        return (x > alpha) * (x - alpha) + (x < -alpha) * (x + alpha)


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    # TODO: implement.
    def duality_gap(self, x):
        ATAx_b = self._f.grad(x)
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        b = self._f.b
        regcoef = self._h.regcoef
        return lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)


class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        return Ax_b.dot(Ax_b) * 0.5 + self.regcoef * np.linalg.norm(x, 1)

    def subgrad(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        return self.matvec_ATx(Ax_b) + self.regcoef * np.sign(x)

    def duality_gap(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        ATAx_b = self.matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self.b, self.regcoef)


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    eps = 1e-13
#    if np.linalg.norm(ATAx_b, np.inf) < eps:
#        mu = Ax_b
#    else:
    mu = Ax_b * min(1, regcoef / np.linalg.norm(ATAx_b, np.inf))
    return Ax_b.dot(Ax_b) * 0.5 + np.linalg.norm(x, 1) * regcoef + mu.dot(mu) * 0.5 + b.dot(mu)


def create_lasso_prox_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)
