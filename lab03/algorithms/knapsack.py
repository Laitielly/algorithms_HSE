from typing import Tuple, List

from numpy.random import randint, random
from random import shuffle
from tqdm import tqdm


class Knapsack:
    def __init__(self, weight: list[int], profit: list[int], capacity: int) -> None:
        '''
        # Параметры:
        - weight - список весов предметов
        - profit - список стоимостей предметов
        - capacity - вместительность рюкзака
        '''
        self.weight = weight
        self.profit = profit
        self.capacity = capacity
        self.n = len(weight)
        self.opt = 0
        self.opt_target = 0

    def fitness(self) -> None:
        self.target = [0 for i in range(self.n_samples)]
        for i in range(self.n_samples):
            current_profit = 0
            current_weight = 0
            x = self.genes[i]
            for j in range(self.n):
                if x & (1 << j):
                    current_profit += self.profit[j]
                    current_weight += self.weight[j]
            if current_weight > self.capacity:
                indexes = [j for j in range(self.n)]
                shuffle(indexes)
                for j in indexes:
                    if x & (1 << j):
                        current_profit -= self.profit[j]
                        current_weight -= self.weight[j]
                        x ^= (1 << j)
                    if current_weight <= self.capacity:
                        break
            if current_profit > self.opt_target:
                self.opt = x
                self.opt_target = current_profit
            self.genes[i] = x
            self.target[i] = current_profit

    def crossover(self, x, y) -> tuple[int, int]:
        i = randint(0, self.n)
        mask1 = (1 << (i+1)) - 1
        mask2 = ((1 << (self.n))-1) ^ mask1
        x_new = (x & mask1) | (y & mask2)
        y_new = (y & mask1) | (x & mask2)
        for i in range(self.n):
            if random() < self.mutation_proba:
                x_new = x_new ^ (1 << i)
            if random() < self.mutation_proba:
                y_new = y_new ^ (1 << i)
        return x_new, y_new

    def selection(self, K: int) -> int:
        indexes = [i for i in range(self.n_samples)]
        shuffle(indexes)
        indexes = indexes[:K]
        mx_i = indexes[0]
        for i in indexes:
            if self.target[i] > self.target[mx_i]:
                mx_i = i
        self.target[mx_i] = -1
        return mx_i

    def calculate(self, n_samples: int, max_iters: int, n_cross: int, mutation_proba: float = 0.01) -> tuple[
        list[int], int]:
        '''
        Функция возвращает список, кодирующий оптимальное решение и значение целевой функции.
        # Параметры:
        - n_samples - количество генов
        - max_iters - максимальное количество иттераций
        - n_cross - количество скрещиваний в каждой иттерации
        - mutation_proba - вероятность мутации
        '''
        self.n_samples = n_samples
        self.mutation_proba = mutation_proba
        max_value = (1 << self.n)
        self.genes = [randint(0, max_value) for i in range(n_samples)]
        for iter in tqdm(range(max_iters)):
            self.fitness()
            for cross in range(n_cross):
                i = self.selection(self.n_samples//4)
                j = self.selection(self.n_samples//4)
                self.genes[i], self.genes[j] = \
                    self.crossover(self.genes[i], self.genes[j])
        self.fitness()
        solution = [0 for i in range(self.n)]
        for i in range(self.n):
            if self.opt & (1 << i):
                solution[i] = 1
        return solution, self.opt_target
