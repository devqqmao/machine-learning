from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List


# Task 0

def gini(x: np.ndarray) -> float:
    probs = np.bincount(x) / len(x)
    return 1 - np.sum(probs ** 2)


def entropy(x: np.ndarray) -> float:
    probs = np.bincount(x) / len(x)
    return -np.sum(probs * np.log2(probs + 1e-16))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion) -> float:
    return (len(left_y) + len(right_y)) * criterion(np.concatenate((left_y, right_y))) - len(left_y) * criterion(
        left_y) - len(right_y) * criterion(right_y)


# Task 1

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


class DecisionTree:

    def __init__(self,
                 X,
                 y,
                 criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1,
                 max_features="auto",
                 ):

        self.root = None

        if criterion == "gini":
            self.criterion = gini
        elif criterion == "entropy":
            self.criterion = entropy
        else:
            self.criterion = entropy

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.n, self.m = X.shape
        self.X_ = X
        self.y_ = y

        n, m = X.shape
        indices = np.arange(n)
        indices_ib = np.random.choice(indices, size=n, replace=True)
        self.indices_oob = np.setdiff1d(indices, indices_ib)

        self.fit(self.X_[indices_ib], self.y_[indices_ib])

    def build_tree(self, indices, depth):

        m = self.X.shape[1]

        if self.max_features == "auto":
            self.features = np.random.choice(np.arange(m), size=int(np.sqrt(m)), replace=False)
        else:
            self.features = np.random.choice(np.arange(m), size=int(self.max_features), replace=False)

        X = self.X[indices, :]
        y = self.y[indices]

        if len(np.unique(y)) == 1:
            return DecisionTreeLeaf(y)
        if depth == self.max_depth:
            return DecisionTreeLeaf(y)

        node_info = {'IG': -float('inf'),
                     "feature": -1,
                     }

        for f in self.features:
            left_bool = X[:, f] == 0
            right_bool = ~left_bool

            left_y = y[left_bool]
            right_y = y[right_bool]

            IG = gain(left_y=left_y, right_y=right_y,
                      criterion=self.criterion)

            if IG > node_info['IG'] and \
                    len(left_y) > self.min_samples_leaf and \
                    len(right_y) > self.min_samples_leaf:
                node_info['IG'] = IG
                node_info['feature'] = f

        if node_info["IG"] == -float('inf'):
            return DecisionTreeLeaf(y)

        left_bool = X[:, node_info['feature']] == 0

        right_bool = ~left_bool

        left = self.build_tree(
            indices[left_bool],
            depth + 1)

        right = self.build_tree(
            indices[right_bool],
            depth + 1)

        return DecisionTreeNode(node_info['feature'], 0, left, right)

    def get_leaf(self, x, node):
        if isinstance(node, DecisionTreeNode):
            dim = node.split_dim
            val = node.split_value
            if x[dim] == val:
                probs = self.get_leaf(x, node.left)
            else:
                probs = self.get_leaf(x, node.right)
        else:
            probs = dict(zip(node.uniques, node.probs))
            return probs
        return probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.X = X
        self.y = y

        self.root = self.build_tree(indices=np.arange(0, self.X.shape[0]), depth=1)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:

        preds = []
        for i in range(X.shape[0]):
            x = X[i, :]
            preds.append(self.get_leaf(x, self.root))
        return preds

    def predict(self, X: np.ndarray) -> list:
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.encoded = False
        if y.dtype == "<U6":
            self.encoded = True
            uniques = np.unique(y)
            self.d = dict(zip(uniques, np.arange(uniques.shape[0])))
            self.d_ = dict(zip(np.arange(uniques.shape[0]), uniques))
            y = np.asarray([self.d[x] for x in y])

        self.trees = []
        cls = DecisionTree(X=X, y=y, criterion=self.criterion, max_depth=self.max_depth,
                           min_samples_leaf=self.min_samples_leaf,
                           max_features=self.max_features)
        self.trees.append(cls)
        for i in range(self.n_estimators):
            cls = DecisionTree(X=X, y=y, criterion=self.criterion, max_depth=self.max_depth,
                               min_samples_leaf=self.min_samples_leaf,
                               max_features=self.max_features)
            self.trees.append(cls)

    def predict(self, X):
        from scipy import stats
        predictions = []
        for i in range(self.n_estimators):
            predictions.append(self.trees[i].predict(X))

        predictions = np.array(predictions).T
        predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
        if self.encoded:
            return np.asarray([self.d_[x] for x in predictions])
        else:
            return predictions

    def oob_err(self):
        from sklearn.metrics import accuracy_score
        scores = []
        scores_perm = []
        differences = []

        for j in range(self.trees[0].m):
            for i in range(len(self.trees)):
                tree = self.trees[i]
                X = np.copy(tree.X_[tree.indices_oob])
                y = np.copy(tree.y_[tree.indices_oob])
                scores.append(accuracy_score(tree.predict(X), y))
                np.random.shuffle(X[:, j])
                scores_perm.append(accuracy_score(tree.predict(X), y))
            differences.append(np.mean(scores) - np.mean(scores_perm))
            scores = []
            scores_perm = []
        return differences


# Task 3

def feature_importance(rfc):
    return rfc.oob_err()


# Task 4

rfc_age = RandomForestClassifier(criterion="gini",
                                 max_depth=15,
                                 min_samples_leaf=10,
                                 max_features=10,
                                 n_estimators=30)

rfc_gender = RandomForestClassifier(criterion="gini",
                                    max_depth=15,
                                    min_samples_leaf=10,
                                    max_features=10,
                                    n_estimators=30)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
from catboost import CatBoostClassifier

catboost_rfc_age = CatBoostClassifier()

catboost_rfc_age.load_model(f'{__file__[:-7]}/age_model.pth')

catboost_rfc_gender = CatBoostClassifier()

catboost_rfc_gender.load_model(f'{__file__[:-7]}/gender_model.pth')
