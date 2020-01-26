#!/usr/bin/python

import numpy as np 
import math

# position [m]
x1_min, x1_max, x1_num = -0.01, 0.05, 61

# speed [m/s]
x2_min, x2_max, x2_num = -.02, 0.04, 61

# angle [rad]
x3_min, x3_max, x3_num = -0.01, 0.01, 21

# angular velocity [rad/s]
x4_min, x4_max, x4_num = -0.005, 0.035, 41

# force [N]
u_min, u_max, u_num = -0.025, 0.175, 41

# sampling time [s]
dt = .05
t_min, t_max = 0, 15
t_num = t_max / dt + 1

dt_lqr = dt
t_num_lqr = t_max / dt_lqr + 1

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

A_lqr = np.matrix([[1,   dt_lqr,         0,          0     ],
                    [0,   1+dt_lqr*a22c,  dt_lqr*a23c,    0     ],
                    [0,   0,          1,          dt_lqr    ],
                    [0,   dt_lqr*a42c,    dt_lqr*a43c,    1     ]])

B_lqr = np.matrix( [0,    dt_lqr*b2c,    0,          dt_lqr*b4c]).reshape((4, 1))

Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
R = np.matrix("10")

# generate states
x1_list = np.linspace(x1_min ,x1_max, x1_num)
x2_list = np.linspace(x2_min ,x2_max, x2_num)
x3_list = np.linspace(x3_min ,x3_max, x3_num)
x4_list = np.linspace(x4_min ,x4_max, x4_num)
u_list  = np.linspace(u_min, u_max, u_num)
t_list  = np.linspace(t_min, t_max, t_num)
t_list_lqr = np.linspace(t_min, t_max, t_num_lqr)

print("x1_list: ", x1_list)
print("x2_list: ", x2_list)
print("x3_list: ", x3_list)
print("x4_list: ", x4_list)
print("u_list: " , u_list)

num_transitions = 0

initial_state = (0.01, 0, -0.01, 0)
end_state = (0, 0, 0, 0)