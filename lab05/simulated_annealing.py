from copy import deepcopy
from random import random
from math import fabs, exp

import numpy as np
from sklearn.cluster import KMeans


class SimulatedAnnealing:
    def __init__(self, data):
        """
        data - матрица: машины х детали
        """
        self.data = np.array(data, dtype=bool)
        self.ones = np.sum(data, axis=1)
        self.opt_target = 0
        self.opt_machine_clusters = None
        self.opt_detail_clusters = None

    def get_submatrix(self, matrix, i, j):
        matrix = np.matrix(matrix)
        i_new, j_new = np.meshgrid(i, j, indexing="ij")
        return matrix[i_new, j_new]

    def get_detail_distribution(self, n_clusters):
        """
        Кластеризует детали алгоритмом KMeans на n_clusters кластеров.
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(self.data.T)
        detail_clusters = [[] for i in range(n_clusters)]
        for i in range(len(kmeans.labels_)):
            detail_clusters[kmeans.labels_[i]].append(i)
        return detail_clusters

    def get_machine_distribution(self, detail_clusters):
        machine_clusters = [[] for i in range(len(detail_clusters))]
        for i in range(len(self.data)):
            labels = [0 for j in range(len(detail_clusters))]
            for j in range(len(detail_clusters)):
                slice = self.data[i][detail_clusters[j]]
                ones_out = self.ones[i] - np.sum(slice)
                zeros_in = len(slice) - np.sum(slice)
                labels[j] = ones_out + zeros_in
            machine_clusters[np.argmin(labels)].append(i)

        for i in range(len(machine_clusters)):
            if len(machine_clusters[i]) == 0:
                for j in range(len(machine_clusters)):
                    if len(machine_clusters[j]) > 1:
                        index = machine_clusters[j].pop()
                        machine_clusters[i].append(index)
                        break

        return machine_clusters

    def get_target(self, machine_clusters, detail_clusters):
        n_1 = np.sum(self.data)
        n_1_in = 0
        n_0_in = 0
        for i in range(len(machine_clusters)):
            slice = self.get_submatrix(
                self.data, machine_clusters[i], detail_clusters[i]
            )
            sum = np.sum(slice)
            n_1_in += sum
            n_0_in += len(machine_clusters[i]) * len(detail_clusters[i]) - sum
        n_1_out = n_1 - n_1_in
        return (n_1 - n_1_out) / (n_1 + n_0_in)

    def singe_move(self, detail_clusters):
        opt_target = 0
        opt_machine_clusters = None
        opt_detail_clusters = None
        for i in range(len(detail_clusters)):
            if len(detail_clusters[i]) > 1:
                for j in range(len(detail_clusters[i])):
                    for k in range(len(detail_clusters)):
                        if i != k:
                            current_detail_clusters = deepcopy(detail_clusters)
                            index = current_detail_clusters[i].pop(j)
                            current_detail_clusters[k].append(index)
                            current_machine_clusters = self.get_machine_distribution(
                                current_detail_clusters
                            )
                            current_target = self.get_target(
                                current_machine_clusters, current_detail_clusters
                            )
                            if opt_target < current_target:
                                opt_target = current_target
                                opt_machine_clusters = current_machine_clusters
                                opt_detail_clusters = current_detail_clusters
        return opt_machine_clusters, opt_detail_clusters, opt_target

    def exchange_move(self, detail_clusters):
        opt_target = 0
        opt_machine_clusters = None
        opt_detail_clusters = None
        indexes = []
        for i in range(len(detail_clusters)):
            for j in range(len(detail_clusters[i])):
                indexes.append((i, j))
        for i in range(len(indexes)):
            for j in range(i+1, len(indexes)):
                if indexes[i][0] != indexes[j][0]:
                    current_detail_clusters = deepcopy(detail_clusters)
                    current_detail_clusters[indexes[i][0]][indexes[i][1]], \
                        current_detail_clusters[indexes[j][0]][indexes[j][1]] = \
                        current_detail_clusters[indexes[j][0]][indexes[j][1]], \
                        current_detail_clusters[indexes[i][0]][indexes[i][1]]
                    current_machine_clusters = self.get_machine_distribution(
                        current_detail_clusters
                    )
                    current_target = self.get_target(
                        current_machine_clusters, current_detail_clusters
                    )
                    if opt_target < current_target:
                        opt_target = current_target
                        opt_machine_clusters = current_machine_clusters
                        opt_detail_clusters = current_detail_clusters
        return opt_machine_clusters, opt_detail_clusters, opt_target

    def is_opt(self, machine_clusters, detail_clusters, target):
        if self.opt_target < target:
            self.opt_target = target
            self.opt_machine_clusters = machine_clusters
            self.opt_detail_clusters = detail_clusters

    def calculate(self, min_clusters, max_clusters, t_0=0.8, a=0.8, threshold=0.05, D=5):
        for i in range(min_clusters, max_clusters+1):
            # print(f'number of clusters: {i}')
            detail_clusters = self.get_detail_distribution(i)
            machine_clusters = self.get_machine_distribution(detail_clusters)
            target = self.get_target(machine_clusters, detail_clusters)
            self.is_opt(machine_clusters, detail_clusters, target)
            t = t_0
            counter = 0
            while t > threshold:
                if counter % D == 0:
                    new_machine_clusters, new_detail_clusters, new_target = self.exchange_move(
                        detail_clusters
                    )
                else:
                    new_machine_clusters, new_detail_clusters, new_target = self.singe_move(
                        detail_clusters
                    )
                counter += 1
                # print(
                #     f'temperature: {t}, old target: {target}, new target: {new_target}'
                # )
                if target < new_target:
                    machine_clusters = new_machine_clusters
                    detail_clusters = new_detail_clusters
                    target = new_target
                    self.is_opt(machine_clusters, detail_clusters, target)
                else:
                    delta = fabs(target - new_target)
                    proba = exp(-delta / t)
                    t *= a
                    if random() < proba:
                        machine_clusters = new_machine_clusters
                        detail_clusters = new_detail_clusters
                        target = new_target

        return self.opt_target, self.opt_machine_clusters, self.opt_detail_clusters
