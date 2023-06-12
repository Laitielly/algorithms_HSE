import numpy as np
from random import shuffle
from algorithms.local_search import LocalSearch
from algorithms.iterated_local_search import IteratedLocalSearch

def decoder_npy(path):
    data = np.load(path, allow_pickle=True)
    d = dict(enumerate(data.flatten(), 1))

    return d[1]

graph = decoder_npy('data/graph.npy')
a = LocalSearch(graph['tai40a']['distance'], graph['tai40a']['stream'])
print(a.calculate(100))

a = IteratedLocalSearch(graph['tai40a']['distance'], graph['tai40a']['stream'])
print(a.calculate(100))