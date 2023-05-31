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


def salesman_report(graphs: dict) -> tuple:
    time_salesman = []
    weight_salesman = []
    way_salesman = []
    answer_salesman = []

    for key in (graphs.keys()):

        start_time = time.time()
        a = TravellingSalesman(graphs[key]['graph'])
        results = a.calculate(150_000, 10, 1_000, 0.5)

        time_salesman.append(time.time() - start_time)
        weight_salesman.append(results[1])
        way_salesman.append(results[0])
        answer_salesman.append(graphs[key]["answer"])

    return (time_salesman, weight_salesman,
            way_salesman, answer_salesman)
