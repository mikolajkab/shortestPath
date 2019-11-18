import numpy as np 
import time
import math

# position [m]
x1_min = -1.2
x1_max = 1.2
x1_num = 21

# speed [m/s]
x2_min = -.55
x2_max = .55
x2_num = 21

# angle [rad]
x3_min = -.25
x3_max = .25
x3_num = 21

# angular velocity [rad/s]
x4_min = -1.5
x4_max = 1.5
x4_num = 21

# force [N]
u_min = -1
u_max = 1
u_num = 21

# sampling time [s]
dt = .25

# matrix coefficients continuous time
a12c = 1
a22c = -0.1818
a23c = 2.673
a42c = -0.4545
a43c = 31.18
b2c  = 1.818
b4c  = 4.545

# matrices discrete time
A = np.array([[1,   dt,         0,          0     ],
              [0,   1+dt*a22c,  dt*a23c,    0     ],
              [0,   0,          1,          dt    ],
              [0,   dt*a42c,    dt*a43c,    1     ]])

B = np.array( [0,    dt*b2c,    0,          dt*b4c]).reshape((4, 1))

print("A: \n", A)
print("B: \n", B)

# generate lists
x1_list = np.linspace(x1_min ,x1_max, x1_num)
x2_list = np.linspace(x2_min ,x2_max, x2_num)
x3_list = np.linspace(x3_min ,x3_max, x3_num)
x4_list = np.linspace(x4_min ,x4_max, x4_num)
u_list  = np.linspace(u_min, u_max, u_num)

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

x1_eq = x1_list[math.floor(len(x1_list)/2)]
x2_eq = x2_list[math.floor(len(x2_list)/2)]
x3_eq = x3_list[math.floor(len(x3_list)/2)]
x4_eq = x4_list[math.floor(len(x4_list)/2)]
u_eq  = u_list[math.floor(len(u_list)/2)]

states, x1_below, x1_above, x2_below, x2_above, x3_below, x3_above, x4_below, x4_above = set(), set(),set(), set(), set(), set(), set(), set(), set()
num_transitions, num_x1_below, num_x1_above, num_x2_below, num_x2_above, num_x3_below, num_x3_above, num_x4_below, num_x4_above = 0, 0, 0, 0, 0, 0, 0, 0, 0

def generateGraphBFS():
    
    time_start = time.time()

    visited = set()
    state_start = (x1_eq, x2_eq, x3_eq, x4_eq)
    queue = [state_start]

    while queue:
        actual = queue.pop(0)
        if actual not in visited:
            visited.add(actual)
            queue.extend(generateAdjacentNodes(actual))

    time_end = time.time()

    # print("number of transitions: ", num_transitions)
    print("number of states: ", len(visited))
    # print("visited states: ", visited)
    # for elem in visited:
    #     print("state: ", elem)
    print("number of state combinations: ", x1_num * x2_num * x3_num * x4_num)
    print("execution time: ", time_end - time_start)

def generateAdjacentNodes(x_k):
    # print("x_k: \n", x_k)

    states = []
    
    for u in u_list:
        x_k = np.array([x_k[0], x_k[1], x_k[2], x_k[3]]).reshape((4, 1))
        x_k_1 = np.add(np.matmul(A, x_k), B * u)

        # print("u: ", u)
        # print("x_k_1: ", x_k_1)

        if (x_k_1[0] >= x1_min and x_k_1[0] <= x1_max) \
            and (x_k_1[1] >= x2_min and x_k_1[1] <= x2_max) \
            and (x_k_1[2] >= x3_min and x_k_1[2] <= x3_max) \
            and (x_k_1[3] >= x4_min and x_k_1[3] <= x4_max):

            x_k_1[0] = min(x1_list, key=lambda x:abs(x-x_k_1[0]))
            x_k_1[1] = min(x2_list, key=lambda x:abs(x-x_k_1[1]))
            x_k_1[2] = min(x3_list, key=lambda x:abs(x-x_k_1[2]))
            x_k_1[3] = min(x4_list, key=lambda x:abs(x-x_k_1[3]))

            # x_k_1 = tuple(x_k_1)
            # print("x_k_1: ", x_k_1)

            if not np.array_equal(x_k_1,x_k):
                states.append(tuple(map(tuple, x_k_1)))
            # num_transitions += 1
            # print("x_k_1: \n", x_k_1)
    
    # for elem in states:
    #     print("state: ", elem)
    return states



def generateGraph():

    states, x1_below, x1_above, x2_below, x2_above, x3_below, x3_above, x4_below, x4_above = set(), set(),set(), set(), set(), set(), set(), set(), set()
    num_transitions, num_x1_below, num_x1_above, num_x2_below, num_x2_above, num_x3_below, num_x3_above, num_x4_below, num_x4_above = 0, 0, 0, 0, 0, 0, 0, 0, 0

    start = time.time()

    for x1 in x1_list:
        for x2 in x2_list:
            for x3 in x3_list:
                for x4 in x4_list:
                    for u in u_list:

                        x_k = np.array([x1, x2, x3, x4]).reshape((4, 1))
                        x_k_1 = np.add(np.matmul(A, x_k), B * u)

                        if x_k_1[0] < x1_min:
                            x1_below.add(tuple(x_k_1[0]))
                            num_x1_below += 1

                        if x_k_1[0] > x1_max:
                            x1_above.add(tuple(x_k_1[0]))
                            num_x1_above += 1

                        if x_k_1[1] < x2_min:
                            x2_below.add(tuple(x_k_1[1]))
                            num_x2_below += 1

                        if x_k_1[1] > x2_max:
                            x2_above.add(tuple(x_k_1[1]))
                            num_x2_above += 1

                        if x_k_1[2] < x3_min:
                            x3_below.add(tuple(x_k_1[2]))
                            num_x3_below += 1

                        if x_k_1[2] > x3_max:
                            x3_above.add(tuple(x_k_1[2]))
                            num_x3_above += 1

                        if x_k_1[3] < x4_min:
                            x4_below.add(tuple(x_k_1[3]))
                            num_x4_below += 1

                        if x_k_1[3] > x4_max:
                            x4_above.add(tuple(x_k_1[3]))
                            num_x4_above += 1

                        if (x_k_1[0] >= x1_min and x_k_1[0] <= x1_max) \
                            and (x_k_1[1] >= x2_min and x_k_1[1] <= x2_max) \
                            and (x_k_1[2] >= x3_min and x_k_1[2] <= x3_max) \
                            and (x_k_1[3] >= x4_min and x_k_1[3] <= x4_max):
                            
                            x_k_1[0] = min(x1_list, key=lambda x:abs(x-x_k_1[0]))
                            x_k_1[1] = min(x2_list, key=lambda x:abs(x-x_k_1[1]))
                            x_k_1[2] = min(x3_list, key=lambda x:abs(x-x_k_1[2]))
                            x_k_1[3] = min(x4_list, key=lambda x:abs(x-x_k_1[3]))
                            
                            states.add(tuple(map(tuple, x_k_1)))
                            num_transitions += 1
                    
    end = time.time()

    print("x_k: \n", x_k)
    print("a: \n", np.matmul(A, x_k))
    print("b: \n", B * u)

    print("number of transitions: ", num_transitions)
    print("number of states: ", len(states))
    print("number of state combinations: ", x1_num * x2_num * x3_num * x4_num)

    print("possible x1 states: ", x1_list)
    print("possible x2 states: ", x2_list)
    print("possible x3 states: ", x3_list)
    print("possible x4 states: ", x4_list)
    print("num x1 below: ", num_x1_below, len(x1_below), ", num x1 above: ", num_x1_above, len(x1_above))
    print("num x2 below: ", num_x2_below, len(x2_below), ", num x2 above: ", num_x2_above, len(x2_above))
    print("num x3 below: ", num_x3_below, len(x3_below), ", num x3 above: ", num_x3_above, len(x3_above))
    print("num x4 below: ", num_x4_below, len(x4_below), ", num x4 above: ", num_x4_above, len(x4_above))

    print("execution time: ", end - start) 

generateGraph()
