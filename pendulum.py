#!/usr/bin/python

import numpy as np 
import time
import math
import scipy.linalg
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_qraph():

    graph = defaultdict(dict)
    time_start = time.time()

    visited = set()
    # queue = [np.matrix([[x1_eq], [x2_eq], [x3_eq], [x4_eq]])]
    queue = [(x1_eq, x2_eq, x3_eq, x4_eq)]

    while queue:
        actual = queue.pop(0)
        # actual_tup = (actual.item(0), actual.item(1), actual.item(2), actual.item(3))

        if actual not in visited:
            visited.add(actual)
            adjacet_nodes, costs = generate_adjacent_nodes(actual)
            queue.extend(adjacet_nodes)

            for node, cost in zip(adjacet_nodes, costs):
                graph[actual][node] = cost

    time_end = time.time()

    print("number of transitions: ", num_transitions)
    print("number of generated states: ", num_generated_states)
    print("number of states: ", len(visited))
    # print("visited states: ", visited)
    # for elem in sorted(visited):
    #     print("state: ", elem)
    print("number of state combinations: ", x1_num * x2_num * x3_num * x4_num)
    print("execution time: ", time_end - time_start)
    for key, val in graph.items():
        print(key, val)

def generate_adjacent_nodes(actual):
    # print("x_k: \n", x_k)

    states = []
    costs = []
    global num_transitions

    x_k = np.matrix([[actual[0]], [actual[1]], [actual[2]], [actual[3]]])

    for u in u_list:
        x_k_1 = np.add(np.matmul(A, x_k), B * u)

        # print("u: ", u)
        # print("x_k_1: ", x_k_1)

        if (x_k_1[0] >= x1_min and x_k_1[0] <= x1_max) \
            and (x_k_1[1] >= x2_min and x_k_1[1] <= x2_max) \
            and (x_k_1[2] >= x3_min and x_k_1[2] <= x3_max) \
            and (x_k_1[3] >= x4_min and x_k_1[3] <= x4_max):

            # round to closes state
            x_k_1[0] = min(x1_list, key=lambda x:abs(x-x_k_1[0]))
            x_k_1[1] = min(x2_list, key=lambda x:abs(x-x_k_1[1]))
            x_k_1[2] = min(x3_list, key=lambda x:abs(x-x_k_1[2]))
            x_k_1[3] = min(x4_list, key=lambda x:abs(x-x_k_1[3]))

            global num_generated_states
            num_generated_states += 1

            # x_k_1 = tuple(x_k_1)
            # print("x_k_1: ", x_k_1)

            # if x_k_1 not in states:
            x_k_1_tup = (x_k_1.item(0), x_k_1.item(1), x_k_1.item(2), x_k_1.item(3))
            
            if x_k_1_tup not in states:
                states.append(x_k_1_tup)
                cost = x_k_1.T*Q*x_k_1 + u.T*R*u
                costs.append(cost.item(0))

                num_transitions += 1

            # print("x_k_1: \n", x_k_1)
    
    # for elem in states:
    #     print("state: ", elem)
    return states, costs

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

def solve_lqr():
    # lqr
    K = dlqr(A,B,Q,R)
    print (K)
    print ("double c[] = {%f, %f, %f, %f};" % (K[0,0], K[0,1], K[0,2], K[0,3]))

    xk = np.matrix(".2 ; .50 ; .1 ; 0")

    X = []
    T = []

    for _ in range(t_list):
        uk = K*xk
        X.append(xk[0,0])
        T.append(xk[2,0])
        xk = A*xk + B*uk

    plt.plot(t_list, X, label="cart position, meters")
    plt.plot(t_list, T, label='pendulum angle, radians')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()




# position [m]
x1_min, x1_max, x1_num = -1.2, 1.2, 9

# speed [m/s]
x2_min, x2_max, x2_num = -.55, .55, 9

# angle [rad]
x3_min, x3_max, x3_num = -.25, .25, 9

# angular velocity [rad/s]
x4_min, x4_max, x4_num = -1.5, 1.5, 9

# force [N]
u_min, u_max, u_num = -1, 1, 11

# sampling time [s]
dt = .25
t_min, t_max = 0, 4
t_num = t_max / dt + 1

# matrix coefficients continuous time
a12c = 1
a22c = -0.1818
a23c = 2.673
a42c = -0.4545
a43c = 31.18
b2c  = 1.818
b4c  = 4.545

# matrices discrete time
A = np.matrix([[1,   dt,         0,          0     ],
              [0,   1+dt*a22c,  dt*a23c,    0     ],
              [0,   0,          1,          dt    ],
              [0,   dt*a42c,    dt*a43c,    1     ]])

B = np.matrix( [0,    dt*b2c,    0,          dt*b4c]).reshape((4, 1))

print("A: \n", A)
print("B: \n", B)

Q = np.matrix("1 0 0 0; 0 .0001 0 0 ; 0 0 1 0; 0 0 0 .0001")
R = np.matrix(".0005")

# generate states
x1_list = np.linspace(x1_min ,x1_max, x1_num)
x2_list = np.linspace(x2_min ,x2_max, x2_num)
x3_list = np.linspace(x3_min ,x3_max, x3_num)
x4_list = np.linspace(x4_min ,x4_max, x4_num)
u_list  = np.linspace(u_min, u_max, u_num)
t_list = np.linspace(t_min, t_max, t_num)

# round states
x1_list = [round(x,10) for x in x1_list]
x2_list = [round(x,10) for x in x2_list]
x3_list = [round(x,10) for x in x3_list]
x4_list = [round(x,10) for x in x4_list]
u_list  = [round(x,10) for x in u_list]

print("x1_list: ", x1_list)
print("x2_list: ", x2_list)
print("x3_list: ", x3_list)
print("x4_list: ", x4_list)
print("u_list: " , u_list)

# find equilibrium state
x1_eq = x1_list[math.floor(len(x1_list)/2)]
x2_eq = x2_list[math.floor(len(x2_list)/2)]
x3_eq = x3_list[math.floor(len(x3_list)/2)]
x4_eq = x4_list[math.floor(len(x4_list)/2)]
u_eq  = u_list[math.floor(len(u_list)/2)]

num_transitions, num_generated_states = 0, 0

generate_qraph()

# solve_lqr()


