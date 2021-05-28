# http://linanqiu.github.io/2018/03/05/Wedding-Seat-Optimization/

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

import numpy as np
import networkx as nx
import pandas as pd
import csv
import sys

i=0
guest_list = []
relationships_edges = {}
with open("/home/matteo/downloads/Matrice per Tavoli - Foglio1.csv", newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        # i+=1
        if not row[0]:
            # print(f"skipping #{i}")
            continue
        guest_list.append(row[0])

with open("/home/matteo/downloads/Matrice per Tavoli - Foglio1.csv", newline='') as csvfile:
    data = csv.reader(csvfile)
    for i, row in enumerate(data):
        if not row[0]:
            # print(f"skipping #{i}")
            continue
        # print(i)
        # print(guest_list[i-1])
        for j, relationship in enumerate(row[i+1:], start=0):
            a = guest_list[i-1]
            b = guest_list[j+i]
            if relationship == 'LV' or relationship == 'L':
                same_table_cost = -75
            elif relationship == 'MD':
                same_table_cost = -100
            elif relationship == 'S':
                same_table_cost = -50
            elif relationship == 'AD':
                same_table_cost = -40
            elif relationship == 'F3' or relationship == 'L3':
                same_table_cost = -30
            elif relationship == 'F2' or relationship == 'L2':
                same_table_cost = -20
            elif relationship == 'F1' or relationship == 'L1':
                same_table_cost = -10
            elif relationship == '':
                same_table_cost = 0
            elif relationship == 'D1':
                same_table_cost = 10
            elif relationship == 'D2':
                same_table_cost = 20
            elif relationship == 'D3' or relationship == 'H':
                same_table_cost = 50
            else:
                print(f"Unknown relationship tag {relationship} for {a}/{b}")
                sys.exit(1)
            relationships_edges[(a, b)] = same_table_cost


table_size = 8
table_count = (len(guest_list)+table_size-1) // table_size

# print(guest_list)

temp_graph = nx.Graph()
for k, v in relationships_edges.items():
    temp_graph.add_edge(k[0], k[1], weight=v)
relationships_mat_unnormed = nx.to_numpy_matrix(temp_graph.to_undirected(), nodelist=guest_list)

relationships_mat = relationships_mat_unnormed / 100

table_seats_initial_array = [[0] * len(guest_list) for i in range(table_count)]
for i in range(len(guest_list)):
    # print(i // table_size)
    table_seats_initial_array[i // table_size][i] = 1

table_seats_matrix = np.matrix(table_seats_initial_array)

# print(table_seats_initial_array)
# print(relationships_mat)

def reshape_to_table_seats(x):
    table_seats = x.reshape(table_count, len(guest_list))
    return table_seats

def cost(x):
    table_seats = reshape_to_table_seats(x)
    table_costs = table_seats * relationships_mat * table_seats.T
    table_cost = np.trace(table_costs)
    return table_cost

def take_step(x):
    table_seats = reshape_to_table_seats(np.matrix(x, copy=True))
    # randomly swap two guests
    table_from, table_to = np.random.choice(table_count, 2, replace=False)

    table_from_guests = np.where(table_seats[table_from] == 1)[1]
    table_to_guests = np.where(table_seats[table_to] == 1)[1]

    table_from_guest = np.random.choice(table_from_guests)
    table_to_guest = np.random.choice(table_to_guests)

    table_seats[table_from, table_from_guest] = 0
    table_seats[table_from, table_to_guest] = 1
    table_seats[table_to, table_to_guest] = 0
    table_seats[table_to, table_from_guest] = 1
    return table_seats

def prob_accept(cost_old, cost_new, temp):
    a = 1 if cost_new < cost_old else np.exp((cost_old - cost_new) / temp)
    return a

def anneal(pos_current, temp=1.0, temp_min=0.00001, alpha=0.9, n_iter=100, audit=False):
    cost_old = cost(pos_current)

    audit_trail = []

    while temp > temp_min:
        for i in range(0, n_iter):
            pos_new = take_step(pos_current)
            cost_new = cost(pos_new)
            p_accept = prob_accept(cost_old, cost_new, temp)
            if p_accept > np.random.random():
                pos_current = pos_new
                cost_old = cost_new
            if audit:
                audit_trail.append((cost_new, cost_old, temp, p_accept))
        temp *= alpha

    return pos_current, cost_old, audit_trail

result = anneal(table_seats_matrix, n_iter=1000)
final_pos = result[0]
# print(final_pos)
# print(final_pos[0,:])
# print(type(result[0]))
for t in range(table_count):
    print("    TABLE")
    for i in range(len(guest_list)):
        # print(final_pos[t,i])
        if final_pos[t,i] > 0:
            print(guest_list[i])
    print()


print(cost(result[0]))
