from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn


# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.centroids = None

    def calculate_distance(self, X):
        return np.apply_along_axis(
            lambda point: min([(np.linalg.norm(point - centroid[0]), centroid[1]) for centroid in self.centroids]),
            axis=1,
            arr=X)

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        self.n, self.m = X.shape[0], X.shape[1]

        # init self.centroids
        if self.init == "random":
            self.fit_random(X, y)
        elif self.init == "sample":
            self.fit_sample(X, y)
        elif self.init == "k-means++":
            self.fit_kmeans(X, y)
        else:  # handling exceptions
            self.fit_random(X, y)

        # fill self.centroids
        prev = np.random.randint(0, self.m, size=(self.n,))
        for _ in range(self.max_iter):

            closest_centroid = self.calculate_distance(X=X)

            # check for convergence
            next = closest_centroid[:, 1]
            if np.allclose(next, prev):
                break
            prev = next

            for i in range(len(self.centroids)):
                closest_indices = list()

                for j in range(len(closest_centroid)):
                    if closest_centroid[j][1] == i:
                        closest_indices.append(j)

                self.centroids[i][0] = np.apply_along_axis(lambda x: np.mean(x), axis=0,
                                                           arr=X[closest_indices])

    def fit_random(self, X, y=None):

        min_max_features = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=X)
        min_max_features = list(zip(min_max_features[0], min_max_features[1]))
        number_of_centroids_with_at_least_1_point = self.n_clusters - 1

        while number_of_centroids_with_at_least_1_point < self.n_clusters:
            self.centroids = [
                [np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=min_max_features), _]
                for _ in range(self.n_clusters)]

            closest_centroid = self.calculate_distance(X=X)
            number_of_centroids_with_at_least_1_point = len(np.unique(closest_centroid[:, 1]))

        # init centroids

    def fit_sample(self, X, y=None):

        indices_of_centroids = np.random.choice(self.n, size=self.n_clusters)
        self.centroids = [[X[indices_of_centroids[_]], _] for _ in range(self.n_clusters)]

    def fit_kmeans(self, X, y=None):

        # init the first centorid
        indices = np.arange(self.n)
        index_of_centroid = np.random.choice(self.n, size=1)[0]
        self.centroids = []

        for i in range(self.n_clusters):
            self.centroids.append([X[index_of_centroid], i])
            indices = np.setdiff1d(indices, index_of_centroid)
            closest_centroids = self.calculate_distance(X=X[indices])

            dist_total = sum(list(map(lambda x: x ** 2, closest_centroids[:, 0])))
            dist_prob = list(map(lambda x: x ** 2 / dist_total, closest_centroids[:, 0]))
            index_of_centroid = np.random.choice(indices, size=1, p=dist_prob)[0]

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        return np.array(list(map(int, self.calculate_distance(X)[:, 1])))


# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps, self.min_samples, self.leaf_size, self.metric = eps, min_samples, leaf_size, metric

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)

        # build the Graph -------------------------

        neighbours = tree.query_radius(X=X, r=self.eps)
        G = {}
        cores = []
        for i, n in enumerate(neighbours):
            if len(n) >= self.min_samples:
                G[i] = n
                cores.append(i)
            else:
                G[i] = None

        # color the graph -----------------------
        from collections import deque

        color = -1
        colors = np.asarray([-1] * len(X))
        for c in cores:
            if colors[c] == -1:
                color += 1
                q = deque()
                q.append(c)

                while q:
                    v = q.popleft()
                    colors[v] = color

                    for u in G[v]:
                        if colors[u] == -1:
                            if G[u] is not None:
                                colors[u] = color
                                q.append(u)
                            else:
                                colors[u] = color

        labels = colors
        return labels


# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        if linkage == 'average':
            self.calc_dist = self.average_link
        elif linkage == 'single':
            self.calc_dist = self.single_link
        elif linkage == 'complete':
            self.calc_dist = self.complete_link

    def average_link(self, a, b):
        return np.nanmean([a, b], axis=0)

    def single_link(self, a, b):
        return np.where(a < b, a, b)  #

    def complete_link(self, a, b):
        return np.where(a >= b, a, b)

    def _calc_dist_mat(self, X, Y):
        dist_mat = (X ** 2).sum(-1).reshape(-1, 1) + (Y ** 2).sum(-1) - 2 * X @ Y.T
        np.fill_diagonal(dist_mat, np.nan)

        return dist_mat

    def remap_labels(self, labels):
        unique = np.unique(labels)
        real = np.arange(len(unique))
        d = dict(zip(unique, real))

        labels = np.vectorize(d.get)(labels)

        return labels

    def upd_dst(self, idx, dist_mat):
        i, j = idx
        new_dst = self.calc_dist(dist_mat[i, :], dist_mat[j, :])
        dist_mat[i, :], dist_mat[:, i] = new_dst, new_dst
        dist_mat[j, :], dist_mat[:, j] = np.nan, np.nan

        np.fill_diagonal(dist_mat, np.nan)

        return dist_mat

    def get_min_id(self, dist_mat):
        idx = np.unravel_index(np.nanargmin(dist_mat, axis=None), dist_mat.shape)

        return idx

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        dist_mat = self._calc_dist_mat(X, X)
        labels = np.arange(len(X))
        c = 0

        while c < len(X) - self.n_clusters:
            i, j = self.get_min_id(dist_mat)
            labels[labels == j] = i
            dist_mat = self.upd_dst((i, j), dist_mat)
            c += 1

        return self.remap_labels(labels)
