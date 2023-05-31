import os
import numpy as np


def decoder_npy(path):
    data = np.load(path, allow_pickle=True)
    d = dict(enumerate(data.flatten(), 1))

    return d[1]


def get_knapsack(path_to_file:str) -> tuple:
    capacity = {}
    weights = {}
    profits = {}
    optimal = {}

    for roots, dirs, files in os.walk(path_to_file):
        for path in files:
            sp_path = path.split('_')

            with open(f'{path_to_file}/{path}', 'r') as file:
                if sp_path[1] == 'c.txt':
                    capacity[sp_path[0]] = [int(line.rstrip()) for line in file][0]

                elif sp_path[1] == 'p.txt':
                    profits[sp_path[0]] = [int(line.rstrip()) for line in file]

                elif sp_path[1] == 'w.txt':
                    weights[sp_path[0]] = [int(line.rstrip()) for line in file]

                elif sp_path[1] == 's.txt':
                    optimal[sp_path[0]] = [int(line.rstrip()) for line in file]

    return capacity, profits, weights, optimal


def get_salesman(path_to_file:str) -> dict:
    graphs = decoder_npy(path_to_file)
    return graphs
