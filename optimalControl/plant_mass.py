#!/usr/bin/python

import numpy as np 
import math

# position [m]
x1_min, x1_max, x1_num = -1, 1, 201

# speed [m/s]
x2_min, x2_max, x2_num = -0.5, 0.5, 101

# force [N]
u_min, u_max, u_num = -0.2, .2, 11

# sampling time [s]
dt = .1
t_min, t_max = 0, 10
t_num = t_max / dt + 1

dt_lqr = dt
t_num_lqr = t_max / dt_lqr + 1

# matrices discrete time
A = np.matrix([[1,   dt],
               [0,   1 ]])

B = np.matrix([0,    dt]).reshape((2, 1))

A_lqr = np.matrix([[1,   dt_lqr],
                   [0,   1     ]])

B_lqr = np.matrix([0,    dt_lqr]).reshape((2, 1))

Q = np.matrix("1 0; 0 1")
R = np.matrix("1")

# generate states
x1_list = np.linspace(x1_min ,x1_max, x1_num)
x2_list = np.linspace(x2_min ,x2_max, x2_num)
u_list  = np.linspace(u_min, u_max, u_num)
t_list  = np.linspace(t_min, t_max, t_num)
t_list_lqr = np.linspace(t_min, t_max, t_num_lqr)

print("x1_list: ", x1_list)
print("x2_list: ", x2_list)
print("u_list: " , u_list)

num_transitions = 0

initial_state = (0.8, -0.1)
end_state = (0, 0)