from numpy.random import randint, random
from random import shuffle
from tqdm import tqdm


class TravellingSalesman:
    def __init__(self, matrix: list[list[int]]) -> None:
        '''
        # Параметры:
        - matrix - весовая матрица графа
        '''
        self.matrix = matrix
        self.n = len(matrix)
        self.opt = [i for i in range(self.n)]
        self.opt_target = 0
        for i in range(self.n):
            self.opt_target += self.matrix[i][(i+1) % self.n]

    def fitness(self):
        self.target = [0 for i in range(self.n_samples)]
        for i in range(self.n_samples):
            for j in range(self.n):
                start = self.genes[i][j]
                end = self.genes[i][(j+1) % self.n]
                self.target[i] += self.matrix[start][end]
            if self.target[i] < self.opt_target:
                self.opt_target = self.target[i]
                self.opt = self.genes[i]

    def selection(self, K: int) -> int:
        indexes = [i for i in range(self.n_samples)]
        shuffle(indexes)
        indexes = indexes[:K]
        mn_i = indexes[0]
        for i in indexes:
            if self.target[i] < self.target[mn_i]:
                mn_i = i
        self.target[mn_i] = (1 << 63) - 1
        return mn_i

    def mutation(self, x: list[int]) -> list[int]:
        for i in range(self.n):
            if random() < self.mutation_proba:
                j = randint(0, self.n)
                while i == j:
                    j = randint(0, self.n)
                x[i], x[j] = x[j], x[i]
        return x

    def permutation_swap(self, x: list[int], y: list[int], p1: int, p2: int) -> list[int]:
        used = set()
        x_new = [0 for i in range(self.n)]
        for i in range(p1, p2+1):
            x_new[i] = y[i]
            used.add(y[i])
        x_cycle = x[p2+1:self.n]+x[0:p2+1]
        x_cycle = [i for i in x_cycle if i not in used]
        i = p2+1
        while i % self.n != p1:
            x_new[i % self.n] = x_cycle[i-p2-1]
            i += 1
        return x_new

    def crossover(self, x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
        p1, p2 = randint(0, self.n, 2)
        if p1 > p2:
            p1, p2 = p2, p1
        x_new = self.permutation_swap(x, y, p1, p2)
        y_new = self.permutation_swap(y, x, p1, p2)
        return x_new, y_new

    def calculate(self, n_samples: int, max_iters: int, n_cross: int,
                  mutation_proba: float = 0.01) -> tuple[list[int], int]:
        '''
        Функция возвращает оптимальную перестановку в виде списка и значение целевой функции.
        # Параметры:
        - n_samples - количество генов
        - max_iters - максимальное количество иттераций
        - n_cross - количество скрещиваний в каждой иттерации
        - mutation_proba - вероятность мутации
        '''
        self.n_samples = n_samples
        self.mutation_proba = mutation_proba
        self.genes = []
        for i in range(n_samples):
            permutation = [i for i in range(self.n)]
            shuffle(permutation)
            self.genes.append(permutation)
        for iter in tqdm(range(max_iters)):
            self.fitness()
            for cross in range(n_cross):
                i = self.selection(self.n_samples//4)
                j = self.selection(self.n_samples//4)
                self.genes[i], self.genes[j] = \
                    self.crossover(self.genes[i], self.genes[j])
                self.genes[i] = self.mutation(self.genes[i])
                self.genes[j] = self.mutation(self.genes[j])
        self.fitness()
        return self.opt, self.opt_target
