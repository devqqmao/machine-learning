import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List
import pandas as pd


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    dataset = pd.read_csv(path_to_csv)
    labels = dataset['label'].replace(['M', 'B'], [1, 0]).to_numpy()
    features = dataset.drop(columns=['label']).to_numpy()
    return features, labels


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    dataset = pd.read_csv(path_to_csv)
    labels = dataset['label'].to_numpy()
    features = dataset.drop(columns=['label']).to_numpy()
    return features, labels


# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    seed = np.random.RandomState(np.random.randint(0, 2 ** 16 - 1))
    train_sz = int(len(y) * ratio)
    seed.shuffle(X)
    seed.shuffle(y)
    X_train, y_train = X[:train_sz], y[:train_sz]
    X_test, y_test = X[train_sz:], y[train_sz:]
    return X_train, y_train, X_test, y_test


# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    num_classes = np.unique(y_true).shape[0]
    precision, recall = np.empty(shape=num_classes), np.empty(shape=num_classes)
    for i in range(num_classes):
        precision[i] = np.shape(np.where((y_pred == i) & (y_true == y_pred)))[1] \
                       / np.shape(np.where((y_pred == i)))[1]
        recall[i] = np.shape(np.where((y_pred == i) & (y_true == y_pred)))[1] \
                    / np.shape(np.where((y_true == i)))[1]
    accuracy = np.shape(np.where((y_true == y_pred)))[1] / y_true.shape[0]
    return precision, recall, accuracy


# Task 4

class Leaf:
    def __init__(self, indices):
        self.indices = indices


class Node:
    def __init__(self, left_node, right_node, median, feature):
        self.left_node = left_node
        self.right_node = right_node
        self.feature = feature
        self.median = median


# Task 4

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.X = X
        self.leaf_size = leaf_size
        self.n, self.m = X.shape

        self.root = self.make_tree(np.arange(0, self.n, 1), 0)

    def make_tree(self, indices, f):
        median = np.median(self.X[indices, f])
        left = np.where(self.X[indices, f] < median)[0]  # returns indices of where, not items
        right = np.where(self.X[indices, f] >= median)[0]  # returns indices of where, not items
        if left.shape[0] >= self.leaf_size and right.shape[0] >= self.leaf_size:
            left_node = self.make_tree(indices[left], (f + 1) % self.m)
            right_node = self.make_tree(indices[right], (f + 1) % self.m)
            return Node(left_node, right_node, median, f)
        return Leaf(indices)

    def find_nn(self, point, node):
        if isinstance(node, Node):
            if point[node.feature] < node.median:
                nearest_neighbours_indices, nearest_neighbours_distances = self.find_nn(point, node.left_node)
                to_visit = node.right_node
            else:
                nearest_neighbours_indices, nearest_neighbours_distances = self.find_nn(point, node.right_node)
                to_visit = node.left_node
        else:
            distances_to_nearest_neighbours = self.get_distance(point, node.indices)
            indices = np.argsort(distances_to_nearest_neighbours)[:self.k]
            return node.indices[indices], distances_to_nearest_neighbours[indices]

        if len(nearest_neighbours_indices) < self.k or nearest_neighbours_distances[-1] > np.abs(
                node.median - point[node.feature]):
            nearest_neighbours_indices_to_visit, nearest_neighbours_distances_to_visit = self.find_nn(point,
                                                                                                      to_visit)
            # merge
            data = list(zip(nearest_neighbours_indices, nearest_neighbours_distances))
            data.extend(zip(nearest_neighbours_indices_to_visit, nearest_neighbours_distances_to_visit))
            data.sort(key=lambda x: x[1])
            nearest_neighbours_indices, nearest_neighbours_distances = list(zip(*data))
            return nearest_neighbours_indices[:self.k], nearest_neighbours_distances[:self.k]
        return np.asarray(nearest_neighbours_indices), np.asarray(nearest_neighbours_distances)

    def get_distance(self, point, nearest_neighbours_indices):
        return np.apply_along_axis(np.linalg.norm, axis=1, arr=point - self.X[nearest_neighbours_indices])

    def query(self, X: np.array, k: int = 1):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        self.k = k
        # np.apply_along_axis(lambda x: print(x), axis=1, arr=X)  # (2, 30)
        return np.apply_along_axis(lambda x: self.find_nn(x, self.root)[0], axis=1, arr=X)


# Осталось реализовать сам классификатор. Реализуйте его, используя KD-дерево.
# Метод __init__ принимает на вход количество соседей, по которым предсказывается класс, и размер листьев KD-дерева.
# Метод fit должен по набору данных и меток строить классификатор.
# Метод predict_proba должен предсказывать веротности классов для заданного набора данных основываясь на классах соседей

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.X = X
        self.y = y
        self.tree = KDTree(self.X, self.leaf_size)
        self.n_classes = len(np.unique(y))

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.

        """

        x_indices = self.tree.query(X, self.n_neighbors)
        y_indices = self.y[x_indices]
        total_probs = list()
        vals = list(map(lambda y: np.unique(y, return_counts=True), y_indices))
        for i in range(len(vals)):
            probs = [0 for _ in range(self.n_classes)]
            for j in range(len(vals[i][0])):
                probs[vals[i][0][j]] = round(vals[i][1][j] / self.n_neighbors, 3)
            total_probs.append(probs)
        return total_probs

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(self.predict_proba(X), axis=1)
