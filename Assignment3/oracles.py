import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        # TODO: Implement for bonus part
        return (self.b.dot(d) - x.dot(self.A.dot(d))) / d.dot(self.A.dot(d))


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

        self.x_f = None
        self.Ax_f = None

        self.x_g = None
        self.Ax_g = None

        self.x_h = None
        self.Ax_h = None

        self.x_fd = None
        self.alpha_f = None
        self.d_f = None
        self.xd_f = None

        self.x_gd = None
        self.alpha_g = None
        self.d_g = None
        self.xd_g = None

        self.Ax_fd = None
        self.Ad_f = None
        self.Axd_f = None

        self.Ax_gd = None
        self.Ad_g = None
        self.Axd_g = None

    def checkeqs(self, x, x1, Ax1, x2, Ax2, x3, Ax3, x4, Ax4, x5, Ax5, x6, Ax6):
        if x1 is not None and (x1 == x).all():
            return Ax1
        elif x2 is not None and (x2 == x).all():
            return Ax2
        elif x3 is not None and (x3 == x).all():
            return Ax3
        elif x4 is not None and (x4 == x).all():
            return Ax4
        elif x5 is not None and (x5 == x).all():
            return Ax5
        elif x6 is not None and (x5 == x).all():
            return Ax6
        else:
            return self.matvec_Ax(x)

    def func(self, x):
        # TODO: Implement
        self.x_f = x
        self.Ax_f = self.checkeqs(x,
                                  self.x_g, self.Ax_g,
                                  self.x_h, self.Ax_h,
                                  self.x_fd, self.Ax_fd,
                                  self.x_gd, self.Ax_gd,
                                  self.xd_f, self.Axd_f,
                                  self.xd_g, self.Axd_g)

        res = np.mean(
            np.logaddexp(np.zeros(self.b.shape), np.multiply(-self.b, self.Ax_f))) + self.regcoef / 2 * np.linalg.norm(
            x) ** 2
        return res

    def grad(self, x):
        # TODO: Implement
        self.x_g = x
        self.Ax_g = self.checkeqs(x,
                                  self.x_f, self.Ax_f,
                                  self.x_h, self.Ax_h,
                                  self.x_fd, self.Ax_fd,
                                  self.x_gd, self.Ax_gd,
                                  self.xd_f, self.Axd_f,
                                  self.xd_g, self.Axd_g)
        """
        if self.x_f is not None and (self.x_f == x).all():
            self.Ax_g = self.Ax_f
        elif self.x_h is not None and (self.x_h == x).all():
            self.Ax_g = self.Ax_h
        else:
            self.Ax_g = self.matvec_Ax(x)
        """
        z = np.multiply(-self.b, self.Ax_g)
        z = scipy.special.expit(z)
        z = np.multiply(z, self.b)
        return -(self.matvec_ATx(z)) / self.b.size + self.regcoef * x

    def hess(self, x):
        # TODO: Implement
        self.x_h = x
        self.Ax_h = self.checkeqs(x,
                                  self.x_f, self.Ax_f,
                                  self.x_g, self.Ax_g,
                                  self.x_fd, self.Ax_fd,
                                  self.x_gd, self.Ax_gd,
                                  self.xd_f, self.Axd_f,
                                  self.xd_g, self.Axd_g)
        """
        if self.x_f is not None and (self.x_f == x).all():
            self.Ax_h = self.Ax_f
        elif self.x_g is not None and (self.x_g == x).all():
            self.Ax_h = self.Ax_g
        else:
            self.Ax_h = self.matvec_Ax(x)
        """
        z = np.multiply(-self.b, self.Ax_h)
        z = scipy.special.expit(z)
        z = np.multiply(z, (np.ones(self.b.shape) - z))
        A1 = self.matmat_ATsA(z) / self.b.size
        A2 = self.regcoef * np.eye(x.size)
        ANS = A1 + A2
        return ANS

    def hess_vec(self, x, v):
        self.x_h = x
        self.Ax_h = self.checkeqs(x,
                                  self.x_f, self.Ax_f,
                                  self.x_g, self.Ax_g,
                                  self.x_fd, self.Ax_fd,
                                  self.x_gd, self.Ax_gd,
                                  self.xd_f, self.Axd_f,
                                  self.xd_g, self.Axd_g)

        z = np.multiply(-self.b, self.Ax_h)
        z = scipy.special.expit(z)
        z = np.multiply(z, (np.ones(self.b.shape) - z))
        z = scipy.sparse.diags(z)
        return self.matvec_ATx(z.dot(self.matvec_Ax(v))) / self.b.size + self.regcoef * v


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        self.d_f = d
        self.x_fd = x
        self.xd_f = x + alpha * d

        if self.x_gd is not None and (self.x_gd == x).all() and \
                self.d_g is not None and (self.d_g == d).all():
            self.Ax_fd = self.Ax_gd
            self.Ad_f = self.Ad_g

        elif self.x_gd is not None and (self.x_gd == x).all():
            self.Ax_fd = self.Ax_gd
            self.Ad_f = self.matvec_Ax(d)

        elif self.x_f is not None and (self.x_f == x).all():
            self.Ax_fd = self.Ax_f
            self.Ad_f = self.matvec_Ax(d)

        elif self.x_g is not None and (self.x_g == x).all():
            self.Ax_fd = self.Ax_g
            self.Ad_f = self.matvec_Ax(d)

        elif self.xd_f is not None and (self.xd_f == x).all():
            self.Ax_fd = self.Axd_f
            self.Ad_f = self.matvec_Ax(d)

        elif self.xd_g is not None and (self.xd_g == x).all():
            self.Ax_fd = self.Axd_g
            self.Ad_f = self.matvec_Ax(d)

        elif self.d_g is not None and (self.d_g == d).all():
            self.Ax_fd = self.matvec_Ax(x)
            self.Ad_f = self.Ad_g
        else:
            self.Ax_fd = self.matvec_Ax(x)
            self.Ad_f = self.matvec_Ax(d)

        self.Axd_f = self.Ax_fd + alpha * self.Ad_f

        res = np.mean(np.logaddexp(np.zeros(self.b.shape), np.multiply(-self.b, self.Axd_f))) + \
              self.regcoef / 2 * np.linalg.norm(self.xd_f) ** 2
        return res

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        self.d_g = d
        self.x_gd = x
        self.xd_g = x + alpha * d
        if self.x_fd is not None and (self.x_fd == x).all() and \
                self.d_f is not None and (self.d_f == d).all():
            self.Ax_gd = self.Ax_fd
            self.Ad_g = self.Ad_f

        elif self.x_fd is not None and (self.x_fd == x).all():
            self.Ax_gd = self.Ax_fd
            self.Ad_g = self.matvec_Ax(d)

        elif self.x_f is not None and (self.x_f == x).all():
            self.Ax_gd = self.Ax_f
            self.Ad_g = self.matvec_Ax(d)

        elif self.x_g is not None and (self.x_g == x).all():
            self.Ax_gd = self.Ax_g
            self.Ad_g = self.matvec_Ax(d)

        elif self.xd_f is not None and (self.xd_f == x).all():
            self.Ax_gd = self.Axd_f
            self.Ad_g = self.matvec_Ax(d)

        elif self.xd_g is not None and (self.xd_g == x).all():
            self.Ax_gd = self.Axd_g
            self.Ad_g = self.matvec_Ax(d)

        elif self.d_f is not None and (self.d_f == d).all():
            self.Ax_gd = self.matvec_Ax(x)
            self.Ad_g = self.Ad_f
        else:
            self.Ax_gd = self.matvec_Ax(x)
            self.Ad_g = self.matvec_Ax(d)

        self.Axd_g = self.Ax_gd + alpha * self.Ad_g
        z = np.multiply(-self.b, self.Axd_g)
        z = scipy.special.expit(z)
        z = np.multiply(z, self.b)
        ans = -z.dot(self.Ad_g) / self.b.size + self.regcoef * (self.xd_g).dot(d)
        return ans

    def hess_vec(self, x, v):
        self.x_h = x
        self.Ax_h = self.checkeqs(x,
                                  self.x_f, self.Ax_f,
                                  self.x_g, self.Ax_g,
                                  self.x_fd, self.Ax_fd,
                                  self.x_gd, self.Ax_gd,
                                  self.xd_f, self.Axd_f,
                                  self.xd_g, self.Axd_g)
        z = np.multiply(-self.b, self.Ax_h)
        z = scipy.special.expit(z)
        z = np.multiply(z, (np.ones(self.b.shape) - z))
        z = scipy.sparse.diags(z)
        return self.matvec_ATx(z.dot(self.matvec_Ax(v))) / self.b.size + self.regcoef * v


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """

    def matvec_Ax(x):
        # TODO: implement proper matrix-vector multiplication
        return A.dot(x)

    def matvec_ATx(x):
        # TODO: implement proper martix-vector multiplication
        return A.T.dot(x)

    def matmat_ATsA(s):
        # TODO: Implement
        return A.T.dot(scipy.sparse.diags(s, 0).dot(A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    # TODO: Implement numerical estimation of the Hessian times vector
    result = np.zeros(x.size)
    e_i = np.zeros(x.shape)
    for i in range(x.size):
        e_i[i] = 1
        result[i] = (func(x + eps * v + eps * e_i) - func(x + eps * v) - func(x + eps * e_i) + func(x)) / (eps ** 2)
        e_i[i] = 0
    return result
