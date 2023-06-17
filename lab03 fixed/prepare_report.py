import time
import pandas as pd
from algorithms.knapsack import Knapsack
from algorithms.travelling_salesman import TravellingSalesman


def res_table_knapsack(columns, result_time, weight, profit, id):
    score_table = pd.DataFrame(columns=columns)

    score_table.loc['Weight', :] = weight
    score_table.loc['Profit', :] = profit
    score_table.loc['Time', :] = result_time
    score_table.loc['ID', :] = id

    return score_table


def get_weight(id:list, weights:list) -> float:
    weight = 0
    for i in range(len(id)):
        weight += (id[i] * weights[i])

    return weight


def res_table_salesman(columns, result_time, weight, true, id):
    score_table = pd.DataFrame(columns=columns)

    score_table.loc['Weight', :] = weight
    score_table.loc['True answer', :] = true
    score_table.loc['Time', :] = result_time
    score_table.loc['Way', :] = id

    return score_table


def knapsack_report(capacity: dict, weights: dict,
                    profits: dict) -> tuple:
    time_knapsack = []
    weight_knapsack = []
    profit_knapsack = []
    id_knapsack = []

    knapsack_keys = sorted(capacity.keys())

    for key in knapsack_keys:

        start_time = time.time()
        a = Knapsack(weights[key], profits[key], capacity[key])


        results = a.calculate(20, 150, 10, 0.3)

        time_knapsack.append(round(time.time() - start_time, 6))
        weight_knapsack.append(get_weight(results[0], weights[key]))
        profit_knapsack.append(results[1])
        id_knapsack.append(results[0])

    return (time_knapsack, weight_knapsack,
            profit_knapsack, id_knapsack)


def salesman_report(graphs: dict, params) -> tuple:

    start_time = time.time()
    a = TravellingSalesman(graphs['graph'])
    results = a.calculate(*params)

    return (*results, graphs['answer'], time.time() - start_time)
