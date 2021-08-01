import numpy as np


class mykmeans:

    def __init__(self, k, c=None, tolerance=1e-3, max_iteration=10000):
        assert k > 1
        self._k = k
        if c:
            assert k == len(
                c
            )
            self._centers = c
        else:
            self._centers = None
        self._tol = tolerance
        self._max_iter = max_iteration

    @property
    def k(self):
        return self._k

    @property
    def centers(self):
        return self._centers

    def fit(self, x):
        if not self._centers:
            self._centers = self._init_center(x, self._k)
        self._centers, labels, best_iter = self._kmeans(x, self._centers,
                                                        self._tol,
                                                        self._max_iter)
        return self._centers, labels, best_iter

    def predict(self, x):
        dist_matrix = np.zeros((len(x), len(self._centers)))
        for idx, center in enumerate(self._centers):
            dist_matrix[:, idx] = self._compute_dist(x, center)
        labels = np.argmin(dist_matrix, axis=1)
        return labels

    def _init_center(self, x, n=2):
        centers = list()
        indexes = list()
        indexes.append(np.random.randint(len(x)))
        center_1 = x[indexes[-1]]
        centers.append(center_1)
        dist_array = self._compute_dist(x, center_1).tolist()
        indexes.append(np.argmax(dist_array))
        centers.append(x[indexes[-1]])
        if n > 2:
            for _ in range(2, n):
                dist_array = np.zeros(len(x))
                for center in centers:
                    dist_array += self._compute_dist(x, center)
                idx = np.argmax(dist_array)
                while (idx in indexes):
                    dist_array[idx] = 0
                    idx = np.argmax(dist_array)
                indexes.append(idx)
                centers.append(x[indexes[-1]])
        return centers

    def _compute_dist(self, x, center):
        return np.sqrt(np.sum(np.square(x - center), axis=1))

    def _l2_norm(self, x):
        x = np.ravel(x, order='K')
        return np.dot(x, x)

    def _kmeans(self, x, centers, tolerance=1e-3, max_iteration=1000):
        labels = np.zeros(len(x))
        n_iteration = 1
        for i in range(max_iteration):
            dist_matrix = np.zeros((len(x), len(centers)))
            for idx, center in enumerate(centers):
                dist_matrix[:, idx] = self._compute_dist(x, center)
            labels = np.argmin(dist_matrix, axis=1)
            new_centers = np.zeros_like(centers)
            for idx in range(len(centers)):
                new_centers[idx] = np.mean(x[labels == idx], axis=0)
            shifts = self._l2_norm(new_centers - centers)
            if shifts <= tolerance:
                break
            else:
                centers = new_centers
                n_iteration = i + 1

        return centers, labels, n_iteration
