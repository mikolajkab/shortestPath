#!/usr/bin/python

import numpy as np 
import time
import math
import scipy.linalg
import matplotlib.pyplot as plt
import graph as gr
from plant_tanks2 import *
from bisect import bisect_left

def generate_qraph():

    graph = gr.Graph()

    visited = set()
    queue = [initial_state]
    while queue:
        actual = queue.pop()
        # print("actual: " , actual)
        if actual not in visited:
            visited.add(actual)
            adjacent_nodes, edges = generate_adjacent_nodes(actual)
            queue.extend(adjacent_nodes)
            for edge in edges:
                graph.add_edge(*edge)

        # print("queue: " , queue)

    # print("states: ", sorted(visited))
    # print("states: ", list(sorted(visited))[0])
    print("number of states: ", len(visited))
    print("number of possible states: ", len(x1_list)*len(x2_list))
    print("number of transitions: ", num_transitions)
    print("end_state in visited: ", end_state in visited)

    return graph

def generate_adjacent_nodes(actual):
    states = []
    edges = []
    global num_transitions

    x1_k = actual[0]
    x2_k = actual[1]

    for u1 in u1_list:
        for u2 in u2_list:

            x1_k1 = c11*x1_k + c12*sqrt(x1_k) + c13*u1
            x2_k1 = c21*x2_k + c22*sqrt(x1_k) + c23*sqrt(x2_k) + c24*u2

            # round to closest state
            x1_k_1 = round(take_closest(x1_list, x1_k1), 5)
            x2_k_1 = round(take_closest(x2_list, x2_k1), 5)

            # check if x_k_1 is within allowed ranges
            if      (x1_k_1 >= x1_min and x1_k_1 <= x1_max) \
                and (x2_k_1 >= x2_min and x2_k_1 <= x2_max):

                x_k_1_tup = (x1_k_1, x2_k_1)

                if x_k_1_tup not in states:

                    x_k_1 = np.matrix([[x1_k_1-end_state[0]], [x2_k_1-end_state[1]]])
                    u = np.matrix([[u1-u_end[0]], [u2-u_end[1]]])

                    cost = x_k_1.T * Q * x_k_1 + u.T * R * u

                    states.append(x_k_1_tup)
                    edges.append((actual, x_k_1_tup, cost.item(0)))
                    num_transitions += 1

    return states, edges

def sqrt(val):
    if val >= 0:
        return math.sqrt(val)
    else:
        return -math.sqrt(abs(val))
    
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

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
    u_1 = []
    u_2 = []

    for _ in range(len(t_list_lqr)):
        uk = K*xk
        x_1.append(xk[0,0])
        x_2.append(xk[1,0])
        u_1.append(uk[0,0])
        u_2.append(uk[1,0])
        xk = A_lqr*xk + B_lqr*uk

    return x_1, x_2, u_1, u_2

time_start = time.time()
graph = generate_qraph()
graph.create_csv()
time_graph_end = time.time()
print("graph generation time: ", time_graph_end - time_start)

path = gr.dijkstra(graph, initial_state, end_state)
time_dijkstra_end = time.time()
print("graph solving time: ", time_dijkstra_end - time_graph_end)

lqr_initial_state = (initial_state[0]-end_state[0], initial_state[1]-end_state[1])
x1_lqr, x2_lqr, u1_lqr, u2_lqr = solve_lqr(lqr_initial_state)

h1_lqr = [i+end_state[0] for i in x1_lqr]
h2_lqr = [i+end_state[1] for i in x2_lqr]

u1_lqr = [i+u_end[0] for i in u1_lqr]
u2_lqr = [i+u_end[1] for i in u2_lqr]

time_lqr_end = time.time()
print("lqr solving time: ", time_lqr_end - time_dijkstra_end)

# print plot
# path = "Route Not Possible"
if path != "Route Not Possible":
    print("path: ", path)
    x1_sp = []
    x2_sp = []

    for node in path:
        x1_sp.append(node[0])
        x2_sp.append(node[1])

    x1_sp.extend([end_state[0]] * (len(t_list) - len(x1_sp)))
    x2_sp.extend([end_state[1]] * (len(t_list) - len(x2_sp)))
    plt.plot(t_list, x1_sp, label="h1 sp [m]")
    plt.plot(t_list, x2_sp, label='h2 sp [m]')

plt.plot(t_list_lqr, h1_lqr, label="h1 lqr [m]]")
plt.plot(t_list_lqr, h2_lqr, label='h2 lqr [m]')
plt.plot(t_list_lqr, u1_lqr, label='u1 lqr [kg/s]')
plt.plot(t_list_lqr, u2_lqr, label='u2 lqr [kg/s]')


plt.legend(loc='upper right')
plt.grid()
plt.show()
