from random import shuffle
from math import sqrt
from tqdm import tqdm
from algorithms.local_search import LocalSearch


class IteratedLocalSearch:
    def __init__(self, distance: list[list[int]], flow: list[list[int]]) -> None:
        self.distance = distance
        self.flow = flow
        self.n = len(distance)
        self.opt_x = None
        self.opt_target = None

    def perturbation(self, x, k):
        '''
        Рандомно перемешивает k элементов перестановки.
        '''
        indexes = [i for i in range(self.n)]
        shuffle(indexes)
        indexes = indexes[:k]
        map = indexes.copy()
        shuffle(map)
        new_x = x.copy()
        for i in range(k):
            new_x[indexes[i]] = x[map[i]]
        return new_x

    def calculate(self, iters) -> tuple[list[int], int]:
        '''
        Алгоритм Iterated local search. 
        Для функции perturbation берется sqrt(n) элементов перестановки.
        '''
        k = int(sqrt(self.n))+1

        x = [i for i in range(self.n)]
        shuffle(x)

        loc_search = LocalSearch(self.distance, self.flow)
        x, target = loc_search.calculate_with_x(x)
        self.opt_x, self.opt_target = x, target

        for i in tqdm(range(iters)):
            x = self.perturbation(x, k)
            x, target = loc_search.calculate_with_x(x)
            if target < self.opt_target:
                self.opt_x = x
                self.opt_target = target

        return self.opt_x, self.opt_target
