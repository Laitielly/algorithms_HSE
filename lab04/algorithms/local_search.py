from random import shuffle


class LocalSearch:
    def __init__(self, distance: list, flow: list) -> None:
        self.distance = distance
        self.flow = flow
        self.n = len(distance)
        self.bits = [0 for i in range(self.n)]

    def calc_target(self, x: list) -> int:
        '''
        Вычисляет целевую функцию
        '''
        target = 0
        for i in range(self.n):
            for j in range(self.n):
                target += self.flow[i][j] * self.distance[x[i]][x[j]]
        return target

    def calc_delta(self, x1: list, x2: list, positions: tuple) -> int:
        '''
        Эквивалетно подсчету self.calc_target(x1)-self.calc_target(x2), если
        x1 и x2 отличаются только в позициях positions[0] и positions[1]
        '''
        delta = 0
        for p in positions:
            for i in range(self.n):
                delta += self.flow[i][p] * self.distance[x1[i]][x1[p]]
                delta -= self.flow[i][p] * self.distance[x2[i]][x2[p]]
                delta += self.flow[p][i] * self.distance[x1[p]][x1[i]]
                delta -= self.flow[p][i] * self.distance[x2[p]][x2[i]]
        return delta

    def first_improvment(self, x: list):
        '''
        first-improvement + don't look bits
        '''
        for i in range(self.n):
            if self.bits[i] == 0:
                for j in range(self.n):
                    if i != j:
                        x_new = x.copy()
                        x_new[i], x_new[j] = x_new[j], x_new[i]
                        if self.calc_delta(x, x_new, (i, j)) > 0:
                            self.bits[j] = 0
                            return x_new
                self.bits[i] = 1
        return x

    def calculate(self) -> tuple:
        x = [i for i in range(self.n)]
        shuffle(x)
        x_new = self.first_improvment(x)
        while x != x_new:
            x = x_new
            x_new = self.first_improvment(x)
        return x, self.calc_target(x)

    def calculate_with_x(self, x: list) -> tuple:
        self.bits = [0 for i in range(self.n)]
        x_new = self.first_improvment(x)
        while x != x_new:
            x = x_new
            x_new = self.first_improvment(x)
        return x, self.calc_target(x)
