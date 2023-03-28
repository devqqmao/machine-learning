import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


class Perceptron:
    def __init__(self, iterations: int = 100):
        self.iters = iterations
        self.w = None

    class Labels:
        def __init__(self, y):
            self.y = y
            self.d_encode = {}
            self.d_decode = {}

    def _encode(self, y):
        self.lbs = self.Labels(y)

    def _convert_labels(self, y, lbs, encode):
        if encode == True:
            unique = np.array([])
            for _ in y:
                if _ not in unique:
                    unique = np.append(unique, _)
                elif unique.size > 1:
                    break
            lbs.d_encode = dict(zip(unique, (-1, 1)))
            lbs.d_decode = dict(zip((-1, 1), unique))

            return np.asarray([lbs.d_encode[x] for x in y])
        else:
            return np.asarray([lbs.d_decode[x] for x in y])

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.n, self.m = X.shape
        self._encode(y)
        self.w = np.zeros(self.m + 1)

        y = self._convert_labels(y, self.lbs, encode=True)
        X = np.column_stack((np.ones(self.n), X))

        for _ in range(self.iters):
            xs = np.asarray([1 if x > 0 else -1 for x in np.sign(X @ self.w) * y])
            idx = np.squeeze(np.argwhere(xs < 0))
            if idx.size > 0:
                idx = np.squeeze(np.random.choice(idx, size=1))  # optimize (any (shuffle))
                self.w += y[idx] * X[idx]
            else:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.column_stack((np.ones(X.shape[0]), X))

        res = np.asarray([1 if x > 0 else -1 for x in X @ self.w])
        y = self._convert_labels(res, self.lbs, encode=False)
        return y


# Task 2


class PerceptronBest:
    def __init__(self, iterations: int = 100):
        self.acc = 0
        self.iters = iterations
        self.w = None

    def _convert_labels(self, y, encode):
        if encode == True:
            # get unique values
            unique = np.array([])
            for _ in y:
                if _ not in unique:
                    unique = np.append(unique, _)
                elif unique.size > 1:
                    break
            self.d_encode = dict(zip(unique, (-1, 1)))
            self.d_decode = dict(zip((-1, 1), unique))

            return np.asarray([self.d_encode[x] for x in y])
        else:
            return np.asarray([self.d_decode[x] for x in y])

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        # init

        self.n, self.m = X.shape
        weights = np.ones(self.m + 1)

        y = self._convert_labels(y, encode=True)
        X = np.column_stack((np.ones(self.n), X))

        for _ in range(self.iters):

            pred_y = np.sign(X @ weights)
            neq = np.where(pred_y != y)[0]

            accuracy = np.mean(pred_y == y)
            if accuracy >= self.acc:
                self.acc = accuracy
                self.w = copy.deepcopy(weights)

            if neq.shape[0] > 0:
                if neq.shape[0] > 1:
                    idx = np.random.randint(0, neq.shape[0])
                else:
                    idx = 0
                weights += y[neq[idx]] * X[neq[idx]]
            else:
                self.w = copy.deepcopy(weights)
                break
        else:
            pred_y = np.sign(X @ weights)
            accuracy = np.mean(pred_y == y)
            if accuracy >= self.acc:
                self.acc = accuracy
                self.w = copy.deepcopy(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.column_stack((np.ones(X.shape[0]), X))
        res = np.sign(X @ self.w)
        y = self._convert_labels(res, encode=False)
        return y


# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    # brightness of an image
    f1 = np.mean(images, axis=(1, 2))
    # symmetrical difference along horizontal axis of an image
    f2 = np.mean(np.abs(images - images[:, ::-1, :]), axis=(1, 2))
    return np.asarray(list(zip(f1, f2)))
