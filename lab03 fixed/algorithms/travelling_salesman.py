from numpy.random import randint, random
from random import shuffle
from tqdm import tqdm


class TravellingSalesman:
    def __init__(self, matrix: list) -> None:
        '''
        # Параметры:
        - matrix - весовая матрица графа
        '''
        self.matrix = matrix
        self.n = len(matrix)
        self.opt = None
        self.opt_target = 1e9

    def local_search_delta(self, x, i, j):
        n = self.n
        m = self.matrix

        if abs(i-j) == 1 or (i == 0 and j == len(x)-1) or (i == len(x)-1 and j == 0):
            if i > j:
                i, j = j, i
            if i == 0 and j == len(x)-1:
                i, j = j, i

            f_new = m[x[(i-1+n) % n]][x[j]] + m[x[j]][x[i]] + m[x[i]][x[(j+1) % n]]

            f_old = m[x[(i-1+n) % n]][x[i]] + m[x[i]][x[j]] + m[x[j]][x[(j+1) % n]]

            return f_old - f_new
        else:
            f_new = m[x[(i-1+n) % n]][x[j]] + m[x[j]][x[(i+1) % n]] +\
                m[x[(j-1+n) % n]][x[i]] + m[x[i]][x[(j+1) % n]]

            f_old = m[x[(i-1+n) % n]][x[i]] + m[x[i]][x[(i+1) % n]] +\
                m[x[(j-1+n) % n]][x[j]] + m[x[j]][x[(j+1) % n]]

            return f_old - f_new

    def local_search(self, x, max_iters=1000):
        bits = [0 for i in range(self.n)]
        for k in range(max_iters):
            flag = False
            for i in range(self.n):
                if bits[i] == 0:
                    for j in range(self.n):
                        if i != j:
                            delta = self.local_search_delta(x, i, j)
                            if delta > 0:
                                x[i], x[j] = x[j], x[i]
                                bits[j] = 0
                                flag = True
                                break
                    if flag:
                        break
                    else:
                        bits[i] = 1
            if not flag:
                break

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

    def mutation(self, x: list) -> list:
        for i in range(self.n):
            if random() < self.mutation_proba:
                j = randint(0, self.n)
                while i == j:
                    j = randint(0, self.n)
                x[i], x[j] = x[j], x[i]
        return x

    def permutation_swap(self, x: list, y: list, p1: int, p2: int) -> list:
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

    def crossover(self, x: list, y: list) -> tuple:
        p1, p2 = randint(0, self.n, 2)
        if p1 > p2:
            p1, p2 = p2, p1
        x_new = self.permutation_swap(x, y, p1, p2)
        y_new = self.permutation_swap(y, x, p1, p2)
        return x_new, y_new

    def calculate(self, n_samples: int, max_iters: int, n_cross: int,
                  mutation_proba: float = 0.01) -> tuple:
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
        self.genes.append([i for i in range(self.n)])
        for i in range(n_samples-1):
            permutation = [i for i in range(self.n)]
            shuffle(permutation)
            self.genes.append(permutation)

        for i in range(self.n_samples):
            self.local_search(self.genes[i])

        for iter in tqdm(range(max_iters)):
            self.fitness()
            for cross in range(n_cross):
                i = self.selection(self.n_samples//4)
                j = self.selection(self.n_samples//4)
                self.genes[i], self.genes[j] = \
                    self.crossover(self.genes[i], self.genes[j])
                self.genes[i] = self.mutation(self.genes[i])
                self.genes[j] = self.mutation(self.genes[j])
                
        for i in range(self.n_samples):
            self.local_search(self.genes[i])
            
        self.fitness()
        return self.opt, self.opt_target
