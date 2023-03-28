import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False


# Task 1

class LinearSVM:
    def __init__(self, C: float):
        self.C = C
        self.support = None
        self.w = None
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, m = X.shape
        y_ = np.outer(y, y)
        H = np.dot(X, X.T)

        P = matrix(y_ * H, tc="d")
        q = matrix(-np.ones(n), tc="d")
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        A = matrix(y.reshape(1, -1), tc="d")
        b = matrix(0.0, tc="d")
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.float64(sol['x'])

        self.support = np.nonzero(alphas)[0]
        self.w = np.zeros(X.shape[1])

        for ind in self.support:
            self.w += alphas[ind] * y[ind] * X[ind]
        self.b = np.mean(y[self.support] - np.dot(X[self.support], self.w))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))


# Task 2

def get_polynomial_kernel(c=1, power=2):
    return lambda x, y: (c + x @ y.T) ** power


def get_gaussian_kernel(sigma=1.):
    return lambda x, y: np.exp(-sigma * np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) ** 2)


# Task 3
class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        self.C = C
        self.X = None
        self.y = None
        self.a = None
        self.kernel = kernel
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.y = np.copy(y)
        kernel = self.kernel(X, X)
        n, m = X.shape
        y_ = np.outer(y, y)

        P = matrix(y_ * kernel, tc="d")
        q = matrix(-np.ones(n), tc="d")
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        A = matrix(y.reshape(1, -1), tc="d")
        y = y.reshape(-1, 1)
        b = matrix(0.0, tc="d")
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        sol = solvers.qp(P, q, G, h, A, b)
        self.a = np.float32(sol['x'])
        self.support = np.where(self.a > 1e-5)[0]
        kernel_ = kernel[:, self.support]
        kernel_ = kernel_[self.support, :]

        self.y = copy.deepcopy(y[self.support])
        self.X = X[self.support]
        self.b = np.mean(y[self.support] - np.sum(self.a[self.support] * y[self.support] * kernel_, axis=1))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self.y = self.y.reshape(-1, 1)
        ans = np.sum(self.a[self.support] * self.y * self.kernel(self.X, X), axis=0)
        return ans + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))
