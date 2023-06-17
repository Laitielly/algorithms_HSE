import os
import time
import numpy as np
import pandas as pd
from simulated_annealing import SimulatedAnnealing


def get_data(paths):
    graphs = {}
    
    for _, _, files in os.walk(paths):
        for f in files:
            total_path = paths + f
           
            data = None
            with open(total_path) as file:
                n, m = map(int, file.readline().split())
                data = [[0 for j in range(m)] for i in range(n)]
                for line in file:
                    s = list(map(int, line.split()))
                    for i in range(1, len(s)):
                        data[s[0]-1][s[i]-1] = 1
            
            graphs[f] = np.array(data)
            
    return graphs


def get_results(data):
    
    cf = []
    times = []
    
    res = {}
    
    for k in data.keys():
        n = (2, 2 * len(data[k]) // 3)
        a = SimulatedAnnealing(data[k])

        start_time = time.time()
        result = a.calculate(*n)
        end_time = round(time.time() - start_time, 2)
        
        cf.append(result[0])
        times.append(end_time)
        
        res[k] = {'cl_m': result[1], 'cl_d': result[2],
                  'num_m': len(data[k]), 'num_d': len(data[k][0])}

    return times, cf, res


def print_clusters(cl):
    for i in cl.keys():
        print(f"In {i}:")
        for m, d in zip(cl[i]['cl_m'], cl[i]['cl_d']):
            print(f"Cluster of machines {', '.join([str(h + 1) for h in m])} makes details "
                  f"{', '.join([str(h + 1) for h in d])}")
        print()


def cluster_to_string(cl, nums):
    res = [''] * nums

    for clust, i in enumerate(cl):
        for j in i:
            res[j] = str(clust + 1)

    return res


def write_to_file(cl):
    for i in cl.keys():
        with open(f"output/{i.split('.')[0]}.sol", 'w', encoding='utf-8') as ff:
            ff.write(' '.join(cluster_to_string(cl[i]['cl_m'], cl[i]['num_m'])))
            ff.write('\n' + ' '.join(cluster_to_string(cl[i]['cl_d'], cl[i]['num_d'])))
      
         
def res_table_ls(columns, result_time, target):
    score_table = pd.DataFrame(columns=columns)

    score_table.loc['Time', :] = result_time
    score_table.loc['Metrics', :] = target

    return score_table


def start(path_data):
    data = get_data(path_data)
    
    result = get_results(data)
    print_clusters(result[2])
    write_to_file(result[2])
    
    return res_table_ls(data.keys(), result[0], result[1])
