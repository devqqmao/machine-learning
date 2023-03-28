import numpy as np
import pandas
import random
import copy


# Task 1
def cyclic_distance(points, dist):
    return np.sum([dist(points[i], points[i + 1]) for i in range(len(points) - 1)]) \
           + dist(points[0], points[-1])


def l2_distance(p1, p2):
    return np.linalg.norm(p1 - p2, ord=2)


def l1_distance(p1, p2):
    return np.linalg.norm(p1 - p2, ord=1)


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist  # Do not change

    def optimize(self, X):
        return self.optimize_explain(X)[-1]

    def find_permutation(self, permutation, dist):
        for j in range(-1, len(permutation) - 1, 1):
            for k in range(-1, len(permutation) - 1, 1):
                if j == k:
                    continue

                el1 = np.copy(permutation[j])
                el2 = np.copy(permutation[k])

                permutation[j] = el2
                permutation[k] = el1

                new_dist = cyclic_distance(self.X[permutation], self.dist)

                if new_dist < dist:
                    return permutation, True
                else:
                    permutation[j] = el1
                    permutation[k] = el2

        return permutation, False

    def optimize_explain(self, X):
        self.X = X
        permutation = np.random.permutation(np.arange(len(X), dtype=int))
        permutations = [permutation]

        for i in range(self.max_iterations - 1):
            dist = cyclic_distance(self.X[permutation], self.dist)
            permutation, flag = self.find_permutation(permutation, dist)
            if flag:
                permutations.append(permutation)
            else:
                break

        return permutations


# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.population = population
        self.survivors = survivors
        self.dist = distance
        # self.iters = iterations
        self.iters = 25

    def optimize(self, X):
        self.X = X
        best_ind = \
            sorted(self.optimize_explain(X)[-1], key=lambda x: cyclic_distance(self.X[x], self.dist), reverse=False)[0]
        return best_ind

    def mutate(self, survivors):
        indices = np.arange(self.survivors, dtype=int)
        population = []
        for i in range(self.population):
            ind1 = survivors[np.random.choice(indices, size=1)[0]]
            ind2 = survivors[np.random.choice(indices, size=1)[0]]
            length = np.random.randint(0, 5, dtype=int)

            split_idx1 = np.random.randint(0, self.X.shape[0], dtype=int)
            split_idx2 = min(split_idx1 + length, self.X.shape[0])

            to_find = ind1[split_idx1:split_idx2]
            to_insert = np.asarray([el for el in ind2 if el in to_find], dtype=int)

            ind3 = np.concatenate((ind1[:split_idx1], to_insert, ind1[split_idx2:]))

            population.append(ind3)
        return population

    def select(self, population):
        survivors = sorted([p for p in population], key=lambda x: cyclic_distance(self.X[x], self.dist),
                           reverse=False)[:self.survivors]

        return survivors

    def optimize_explain(self, X):
        self.X = X
        population = [np.random.permutation(np.arange(len(X), dtype=int)) for _ in range(self.population)]
        epochs = [population]

        for i in range(self.iters - 1):
            survivors = self.select(population)
            population = self.mutate(survivors)

            epochs.append(population)

        return epochs


# Task 4

class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """

        self.X = X
        self.voc_limit = voc_limit
        self.vocab = {}

        for sentence in self.X:
            sentence = ''.join([s for s in sentence if s not in r"[,.?“/!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”*°°*`)(]"])
            for word in sentence.split(' '):
                if word.lower() in self.vocab:
                    self.vocab[word.lower()] += 1
                else:
                    self.vocab[word.lower()] = 1

        lb = int(0.0025 * len(self.vocab))
        ub = self.voc_limit + lb

        self.vocab = {item[0]: (idx, item[1]) for idx, item in
                      enumerate(sorted(self.vocab.items(), key=lambda item: item[1], reverse=True)[lb:ub])}

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            который необходимо векторизовать.
        
        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """

        def vectorize(sentence):
            sentence = ''.join([s for s in sentence if s not in r"[,.?“/!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”*°°*`)(]"])
            vector = np.zeros(self.voc_limit)
            for word in sentence.split(' '):
                word = word.lower()
                if word in self.vocab:
                    vector[self.vocab[word][0]] += 1
            return vector

        return np.asarray([vectorize(s) for s in X])


# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        m, n = X.shape
        self.prior = np.zeros(self.n_classes)
        self.prob = np.zeros((self.n_classes, n))
        for i, c in enumerate(self.classes):
            X_by_class = X[y == c]
            self.prior[i] = np.log(X_by_class.shape[0] / m)
            self.prob[i] = np.log((X_by_class.sum(axis=0) + self.alpha) / (np.sum(X_by_class) + self.alpha * n))

    def predict(self, X: np.ndarray) -> list:
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes))
        for i in range(self.n_classes):
            log_proba[:, i] = self.prior[i] + np.sum(self.prob[i, :] * X, axis=1)
        return log_proba
