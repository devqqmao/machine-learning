import numpy as np


# Task 1

def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    return np.mean((y_true - y_predicted) ** 2)


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    return 1 - (np.sum((y_true - y_predicted) ** 2) / (np.sum((np.mean(y_true) - y_true) ** 2)))


# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.column_stack((X, np.ones(len(X))))
        self.weights = np.linalg.pinv(X) @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.column_stack((X, np.ones(len(X))))
        return X @ self.weights


# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations: int = 10000, l: float = 0.):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_points = X.shape[0]
        bias = np.ones(n_points)
        X = np.column_stack((bias, X))
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)

        for it in range(self.iterations):
            grad = 2 / X.shape[0] * (X.T @ (X @ self.weights - y)) + self.l * np.sign(self.weights)
            self.weights -= self.alpha * grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        bias = np.ones(X.shape[0])
        X = np.column_stack((bias, X))
        return X @ self.weights


# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[1:])


def get_most_important_features(linear_regression):
    features = get_feature_importance(linear_regression)
    sorted_features = sorted(np.copy(features), reverse=True)
    return np.asarray([np.squeeze(np.where(features == x)) for x in sorted_features])


def get_most_important_features2(linear_regression):
    features = get_feature_importance(linear_regression)
    sorted_features = sorted(enumerate(features), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_features]
