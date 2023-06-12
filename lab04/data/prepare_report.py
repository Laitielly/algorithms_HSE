import time
import numpy as np
import pandas as pd
from algorithms.local_search import LocalSearch
from algorithms.iterated_local_search import IteratedLocalSearch


def decoder_npy(path):
    data = np.load(path, allow_pickle=True)
    d = dict(enumerate(data.flatten(), 1))

    return d[1]


def get_local_search(graph, keys):
    times = []
    way = []
    targets = []

    for i in keys:
        a = LocalSearch(graph[i]['distance'], graph[i]['stream'])

        start_time = time.time()
        result = a.calculate()
        end_time = round(time.time() - start_time, 4)

        way.append(result[0])
        targets.append(result[1])
        times.append(end_time)

    return times, targets, way


def get_ILS(graph, keys):
    times = []
    way = []
    targets = []

    for i in keys:
        a = IteratedLocalSearch(graph[i]['distance'], graph[i]['stream'])

        start_time = time.time()
        result = a.calculate(100)
        end_time = round(time.time() - start_time, 4)

        way.append(result[0])
        targets.append(result[1])
        times.append(end_time)

    return times, targets, way


def res_table_ls(type_alg, columns, result_time, target, true_target):
    score_table = pd.DataFrame(columns=columns)

    score_table.loc[type_alg, :] = ['-'] * len(columns)
    score_table.loc['Time', :] = result_time
    score_table.loc['Target', :] = target
    score_table.loc['True Target', :] = true_target

    return score_table


def print_way(way, columns, typ_alg):
    for way, f in zip(way, columns):
        result_string = ''
        print(f"{f}:")
        for i in way:
            result_string += f"{i} "
            print(f"{i} -> ", end='')
        print(way[0])
        print()

        write_to_file(f, result_string, typ_alg)


def write_to_file(filename, way, typ_alg):
    with open(f"{filename}_{typ_alg}.sol", 'w', encoding='utf-8') as ff:
        ff.write(way)
