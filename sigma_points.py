import numpy as np
from scipy.linalg import cholesky
from math import sqrt


class MerweScaledSigmaPoints:
    """
    Van der Merwe (2004) scaled sigma points.
    Generates 2n+1 points.

    Parameters
    ----------
    n     : state dimension
    alpha : spread around the mean (typically 1e-3)
    beta  : prior knowledge of distribution (2 for Gaussian)
    kappa : secondary scaling (typically 0)
    """

    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.sqrt = cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self._compute_weights()

    def num_sigmas(self):
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        n = self.n
        if np.isscalar(x):
            x = np.asarray([x])
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n) * P)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1]     = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x,  U[k])
        return sigmas

    def _compute_weights(self):
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        c = 0.5 / (n + lambda_)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)


class JulierSigmaPoints:
    """
    Julier & Uhlmann (1997) sigma points.
    Generates 2n+1 points. Mean and covariance weights are identical.

    Parameters
    ----------
    n     : state dimension
    kappa : scaling factor (0 for standard; 3-n minimises 4th-order errors)
    """

    def __init__(self, n, kappa=0., sqrt_method=None, subtract=None):
        self.n = n
        self.kappa = kappa
        self.sqrt = cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self._compute_weights()

    def num_sigmas(self):
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        n = self.n
        if np.isscalar(x):
            x = np.asarray([x])
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        U = self.sqrt((n + self.kappa) * P)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1]     = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x,  U[k])
        return sigmas

    def _compute_weights(self):
        n = self.n
        self.Wm = np.full(2 * n + 1, 0.5 / (n + self.kappa))
        self.Wm[0] = self.kappa / (n + self.kappa)
        self.Wc = self.Wm.copy()


class SimplexSigmaPoints:
    """
    Moireau & Chapelle simplex sigma points.
    Generates n+1 points (minimal set).

    Parameters
    ----------
    n     : state dimension
    alpha : scaling factor (default 1)
    """

    def __init__(self, n, alpha=1., sqrt_method=None, subtract=None):
        self.n = n
        self.alpha = alpha
        self.sqrt = cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self._compute_weights()

    def num_sigmas(self):
        return self.n + 1

    def sigma_points(self, x, P):
        n = self.n
        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        U = self.sqrt(P)

        lambda_ = n / (n + 1)
        Istar = np.array([[-1 / sqrt(2 * lambda_), 1 / sqrt(2 * lambda_)]])

        for d in range(2, n + 1):
            row = np.ones((1, Istar.shape[1] + 1)) / sqrt(lambda_ * d * (d + 1))
            row[0, -1] = -d / sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros(Istar.shape[0])], row]

        scaled_unitary = U.T @ (sqrt(n) * Istar)
        sigmas = self.subtract(x, -scaled_unitary)
        return sigmas.T   # shape: (n+1, n)

    def _compute_weights(self):
        c = 1.0 / (self.n + 1)
        self.Wm = np.full(self.n + 1, c)
        self.Wc = self.Wm.copy()


class SphericalRadialSigmaPoints:
    """
    Arasaratnam & Haykin (2009) spherical-radial cubature points.
    Generates 2n points with equal weights.

    Reference: "Cubature Kalman Filters", IEEE TAC 2009.
    """

    def __init__(self, n):
        self.n = n
        self._compute_weights()

    def num_sigmas(self):
        return 2 * self.n

    def sigma_points(self, x, P):
        n = self.n
        if np.isscalar(x):
            x = np.asarray([x])
        P = np.atleast_2d(P)

        U = cholesky(P) * sqrt(n)   # scaled upper-triangular factor

        sigmas = np.empty((2 * n, n))
        for k in range(n):
            sigmas[k]     = x + U[k]
            sigmas[n + k] = x - U[k]
        return sigmas

    def _compute_weights(self):
        w = 1.0 / (2 * self.n)
        self.Wm = np.full(2 * self.n, w)
        self.Wc = self.Wm.copy()
