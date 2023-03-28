from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List


# Task 1

def gini(x: np.ndarray) -> float:
    uniques, counts = np.unique(x, return_counts=True)
    counts = counts / len(x)
    # return np.sum([c * (1 - c) for c in counts])
    return 1 - np.sum([c ** 2 for c in counts])


def entropy(x: np.ndarray) -> float:
    uniques, counts = np.unique(x, return_counts=True)
    counts = counts / len(x)
    return np.sum([-1 * c * np.log2(c) for c in counts])


# def gini(x: np.ndarray) -> float:
#     probs = np.bincount(x) / len(x)
#     return 1 - np.sum(probs ** 2)
#
#
# def entropy(x: np.ndarray) -> float:
#     probs = np.bincount(x) / len(x)
#     return -np.sum(probs * np.log2(probs + 1e-16))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    return (len(left_y) + len(right_y)) * criterion(np.concatenate((left_y, right_y))) - len(left_y) * criterion(
        left_y) - len(right_y) * criterion(right_y)


# Task 2

class DecisionTreeLeaf:

    def __init__(self, ys):
        uniques, counts = np.unique(ys, return_counts=True)
        self.y = uniques[np.argmax(counts)]
        self.uniques = uniques
        self.probs = counts / sum(counts)


class DecisionTreeNode:

    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


# Task 3

class DecisionTreeClassifier:
    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        self.root = None

        if criterion == "gini":
            self.criterion = gini
        elif criterion == "entropy":
            self.criterion = entropy
        else:
            self.criterion = entropy

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # UDFs

    def build_tree(self, indices, depth):

        # constraints:
        # 1. max_depth
        # 2. min_samples_leaf => calculate inside # get from y
        # 3. all objects of node belong to the same class => make a leaf

        # algorithm:
        # 1. while (recursion inside every function go through)
        # Nf перебираем все признаки
        # Перебираем все значения признака, их максимум Ns
        # Разбиваем наше множество (за линию) + считаем information gain Ns + Ns

        X = self.X[indices]
        y = self.y[indices]

        if len(np.unique(y)) == 1:
            return DecisionTreeLeaf(y)
        if depth == self.max_depth:
            return DecisionTreeLeaf(y)

        node_info = {'IG': -float('inf'),
                     "feature": -1,
                     "value": -1}

        for i in range(X.shape[1]):
            for j in range(len(indices)):

                left_bool = X[:, i] < X[j][i]
                right_bool = ~left_bool

                left_y = y[left_bool]
                right_y = y[right_bool]

                IG = gain(left_y=left_y, right_y=right_y,
                          criterion=self.criterion)

                if IG > node_info['IG'] and \
                        len(left_y) > self.min_samples_leaf and \
                        len(right_y) > self.min_samples_leaf:
                    node_info['IG'] = IG
                    node_info['feature'] = i
                    node_info['value'] = X[j][i]

        if node_info["IG"] == -float('inf'):
            return DecisionTreeLeaf(y)

        # left
        left_bool = X[:, node_info['feature']] < node_info['value']
        # right
        right_bool = ~left_bool

        left = self.build_tree(
            indices[left_bool],
            depth + 1)

        right = self.build_tree(
            indices[right_bool],
            depth + 1)

        return DecisionTreeNode(node_info['feature'], node_info['value'], left, right)

    def get_leaf(self, x, node):
        if isinstance(node, DecisionTreeNode):
            dim = node.split_dim
            val = node.split_value
            if x[dim] < val:
                probs = self.get_leaf(x, node.left)
            else:
                probs = self.get_leaf(x, node.right)
        else:
            probs = dict(zip(node.uniques, node.probs))
            return probs
        return probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        # speed enhancements:
        # take random \sqrt(n) features

        n, m = X.shape
        if m > 2 ** 6:
            self.features = np.random.choice(np.arange(m), size=int(np.sqrt(m)), replace=False)
        else:
            self.features = np.arange(m)

        self.X = X[:, self.features]
        self.y = y

        self.root = self.build_tree(indices=np.arange(0, n), depth=1)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """

        preds = []
        X = X[:, self.features]
        for x in X:
            preds.append(self.get_leaf(x, self.root))
        return preds

    def predict(self, X: np.ndarray) -> list:
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


# Task 4
task4_dtc = DecisionTreeClassifier(max_depth=6, min_samples_leaf=12, criterion="gini")
