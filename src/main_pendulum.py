#!/usr/bin/python

import numpy as np 
import time
import math
import scipy.linalg
import matplotlib.pyplot as plt
import graph as gr
from plant_pendulum import *
import pickle

def generate_qraph():

    graph = gr.Graph()

    visited = set()
    queue = [initial_state]
    while queue:
        actual = queue.pop(0)
        if actual not in visited:
            visited.add(actual)
            adjacent_nodes, edges = generate_adjacent_nodes(actual)
            queue.extend(adjacent_nodes)
            for edge in edges:
                graph.add_edge(*edge)

    print("number of states: ", len(visited))
    print("number of transitions: ", num_transitions)
    print("end_state in visited: ", end_state in visited )

    return graph

def generate_adjacent_nodes(actual):
    states = []
    edges = []
    global num_transitions

    x_k = np.matrix([[actual[0]], [actual[1]], [actual[2]], [actual[3]]])

    for u in u_list:
        x_k_1 = A * x_k + B * u

        # round to closest state
        x_k_1[0] = round(x_k_1.item(0), 3)
        x_k_1[1] = round(x_k_1.item(1), 3)
        x_k_1[2] = round(x_k_1.item(2), 3)
        x_k_1[3] = round(x_k_1.item(3), 3)

        # check if x_k_1 is within allowed ranges
        if      (x_k_1[0] >= x1_min and x_k_1[0] <= x1_max) \
            and (x_k_1[1] >= x2_min and x_k_1[1] <= x2_max) \
            and (x_k_1[2] >= x3_min and x_k_1[2] <= x3_max) \
            and (x_k_1[3] >= x4_min and x_k_1[3] <= x4_max):

            x_k_1_tup = (x_k_1.item(0), x_k_1.item(1), x_k_1.item(2), x_k_1.item(3))

            # if x_k_1_tup not in states:
            states.append(x_k_1_tup)
            cost = x_k_1.T * Q * x_k_1 + u.T * R * u
            edges.append((actual, x_k_1_tup, cost.item(0)))
            num_transitions += 1

    return states, edges

def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    return -K

def solve_lqr(initial_state):
    K = dlqr(A_lqr,B_lqr,Q,R)

    xk = np.matrix(initial_state).transpose()

    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []
    u_k = []

    for _ in range(len(t_list_lqr)):
        uk = K*xk
        x_1.append(xk[0,0])
        x_2.append(xk[1,0])
        x_3.append(xk[2,0])
        x_4.append(xk[3,0])
        u_k.append(uk[0,0])
        xk = A_lqr*xk + B_lqr*uk

    return x_1, x_2, x_3, x_4, u_k

# time_start = time.time()
# graph = generate_qraph()
# time_graph_end = time.time()
# print("graph generation time: ", time_graph_end - time_start)

# path = gr.dijkstra(graph, initial_state, end_state)
# time_dijkstra_end = time.time()
# print("graph solving time: ", time_dijkstra_end - time_graph_end)

# x_1_lqr, x_2_lqr, x_3_lqr, x_4_lqr, u_k_lqr = solve_lqr(initial_state)

# # print plot
# x_1_sp = []
# x_2_sp = []
# x_3_sp = []
# x_4_sp = []

# for node in path:
#     x_1_sp.append(node[0])
#     x_2_sp.append(node[1])
#     x_3_sp.append(node[2])
#     x_4_sp.append(node[3])

# x_1_sp.extend([0] * (len(t_list) - len(x_1_sp)))
# x_2_sp.extend([0] * (len(t_list) - len(x_2_sp)))
# x_3_sp.extend([0] * (len(t_list) - len(x_3_sp)))
# x_4_sp.extend([0] * (len(t_list) - len(x_4_sp)))

# plt.plot(t_list_lqr, x_1_lqr, label="x1 lqr [m]")
# plt.plot(t_list_lqr, x_2_lqr, label='x2 lqr [rad]')
# plt.plot(t_list_lqr, x_3_lqr, label="x3 lqr [m/s]")
# plt.plot(t_list_lqr, x_4_lqr, label='x4 lqr [rad/s]')
# plt.plot(t_list_lqr, u_k_lqr, label='u lqr [N]')
# plt.plot(t_list, x_1_sp, label="x1 sp [m]")
# plt.plot(t_list, x_2_sp, label='x2 sp [m/s]')
# plt.plot(t_list, x_3_sp, label="x3 sp [rad]")
# plt.plot(t_list, x_4_sp, label='x4 sp [rad/s]')

# plt.legend(loc='upper right')
# plt.grid()
# plt.show()  

print(A)
print(B)
print(Q)
print(R)