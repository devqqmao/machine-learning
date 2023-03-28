import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.w = np.random.randn(in_features, out_features) / ((in_features + out_features) ** 0.5)
        self.b = np.random.randn(out_features)
        self.w_grad = None
        self.b_grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) < 2:
            x = x.reshape(1, -1)
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, d: np.ndarray) -> np.ndarray:
        if len(d.shape) < 2:
            d = d.reshape(1, -1)

        self.w_grad = np.dot(self.x.T, d)
        self.b_grad = d.sum(axis=0)
        return np.dot(d, self.w.T)

    def update(self, alpha: float) -> NoReturn:
        self.w -= alpha * self.w_grad
        self.b -= alpha * self.b_grad


class ReLU(Module):

    def __init__(self):
        self.new_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            x = x.reshape(1, -1)
        self.x = x
        return np.maximum(0, x)

    def backward(self, d) -> np.ndarray:
        return np.where(self.x > 0, 1, 0) * d


# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def SoftMax(self, x):
        e = np.exp(x)
        e_sum = np.sum(e, axis=1)
        return e / e_sum[:, np.newaxis]

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:

        for _ in range(self.epochs):
            indices = list(range(len(X)))
            np.random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                f = indices[i: i + self.batch_size]

                X_batch = X[f]
                y_batch = np.array(y)[f]

                prob = self.predict_proba(X_batch)
                prob[range(min(len(indices) - i, self.batch_size)), y_batch] -= 1

                for layer in reversed(self.modules):
                    prob = layer.backward(prob)
                    layer.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x = X
        for l in self.modules:
            x = l.forward(x)
        return self.SoftMax(x)

    def predict(self, X) -> np.ndarray:
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


# Task 3

classifier_moons = MLPClassifier([Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 2)])
classifier_blobs = MLPClassifier([Linear(2, 8), ReLU(), Linear(8, 8), ReLU(), Linear(8, 3)])


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_model(self):
        self.load_state_dict(torch.load(f'{__file__[:-7]}/model.pth'))

    def save_model(self):
        torch.save(self.state_dict(), 'model.pth')


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    criterion = nn.CrossEntropyLoss()
    pre = model(X)
    loss = criterion(pre, y)
    return loss
