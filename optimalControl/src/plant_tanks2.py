#!/usr/bin/python

import numpy as np 
import math

# input data [m]
A1, A2, Ap1, Ap2, ro, g, dt = .5, 1.5, .1, .1, .5, 9.81, 0.03

# initial and end states
initial_state = (0.5, 0.7)
end_state = (.7, .9)
u_end = (ro*Ap1*math.sqrt(2*g*end_state[0]), ro*Ap2*math.sqrt(2*g*end_state[1])-ro*Ap1*math.sqrt(2*g*end_state[0])) 

# h1 [m]
x1_min, x1_max = initial_state[0]+0, initial_state[0]+.22

# h2 [m]
x2_min, x2_max = initial_state[1]+0, initial_state[1]+.22

x11 = np.linspace(initial_state[0]+0, initial_state[0]+0.14, 8)
x12 = np.linspace(initial_state[0]+0.155, initial_state[0]+.220, 14)
x1_list = np.concatenate((x11 ,x12))
x21 = np.linspace(initial_state[1]+0, initial_state[1]+0.1, 11)
x22 = np.linspace(initial_state[1]+0.11, initial_state[1]+.22, 12)
x2_list = np.concatenate((x21 ,x22))

u11 = np.linspace(0, .1, 6)
u12 = np.linspace(.15, .6, 10)
u1_list = np.concatenate((u11, u12))
u21 = np.linspace(0, .2, 11)
u22 = np.linspace(.25, .6, 8)
u2_list = np.concatenate((u21, u22))

u1_list = [round(i,4) for i in u1_list]
u2_list = [round(i,4) for i in u2_list]

# nonlinear discrete equations:
# x1(k+1) = x1(k) - dt*Ap1/A1*sqrt(2*g)*sqrt(x1(k)) + dt/(ro*A1)*u1(k)
# x2(k+1) = x2(k) + dt*Ap2/A2*sqrt(2*g)*sqrt(x1(k)) - dt*Ap2/A2*sqrt(2*g)*sqrt(x2(k)) + dt/(ro*A2)*u2(k)
# with coefficients
# x1(k+1) = c11*x1(k) + c12*sqrt(x1(k)) + c13*u1(k)
# x2(k+1) = c21*x2(k) + c22*sqrt(x1(k)) + c23*sqrt(x2(k)) + c24*u2(k)

c11 =  1
c12 = -dt*Ap1/A1*math.sqrt(2*g)
c13 =  dt/(ro*A1)
c21 =  1
c22 =  dt*Ap1/A2*math.sqrt(2*g)
c23 = -dt*Ap2/A2*math.sqrt(2*g)
c24 =  dt/(ro*A2)

print("c11: ", c11)
print("c12: ", c12)
print("c13: ", c13)
print("c21: ", c21)
print("c22: ", c22)
print("c23: ", c23)
print("c24: ", c24)

# linear continuous equation x_dot = Ac*x + Bc*u 
# around h10 = x10, h20 = x20
x10 = .6
x20 = .8
a11c = -Ap1/A1*g/math.sqrt(2*g*x10)
a12c = 0
a21c = Ap1/A2*g/math.sqrt(2*g*x10)
a22c = -Ap2/A2*g/math.sqrt(2*g*x20)
b11c = 1/(ro*A1)
b12c = 0
b21c = 0
b22c = 1/(ro*A2)

# continuous matrices
Ac = np.matrix([[a11c,   a12c],
                [a21c,   a22c]])

Bc = np.matrix([[b11c,   b12c],
                [b21c,   b22c]])

# linear discrete x_k_1 = Ad*x_k + Bd*u_k
# around h10 = x10, h20 = x20
a11d = 1+dt*a11c
a12d = 0
a21d = dt*a21c
a22d = 1+dt*a22c
b11d = dt*b11c
b12d = 0
b21d = 0
b22d = dt*b22c

# matrices discrete
A = np.matrix([[a11d,   a12d],
               [a21d,   a22d]])

B = np.matrix([[b11d,   b12d],
               [b21d,   b22d]])

A_lqr, B_lqr = A, B

Q = np.matrix("1 0; 0 1")
R = np.matrix(".1 0; 0 .1")

# sampling time [s]
t_min, t_max = 0, 2
t_num = t_max / dt + 1

t_list  = np.linspace(t_min, t_max, t_num)
t_list_lqr = t_list

num_transitions = 0

print("Ac: ", Ac)
print("Bc: ", Bc)
print("x1_list: ", x1_list)
print("x2_list: ", x2_list)
print("u1_list: ", u1_list)
print("u2_list: ", u2_list)
