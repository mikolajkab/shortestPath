#!/usr/bin/python

import numpy as np 
import math

# input data [m]
A1, A2, A3, Ap1, Ap2, Ap3, ro, g, dt = .5, 1.5, 1, .1, .1, .1, .5, 9.81, 0.03
initial_state = (.4, .5, .6)
end_state = (.6, .7, .8)

u_end = (ro*Ap1*math.sqrt(2*g*end_state[0]), \
         ro*Ap2*math.sqrt(2*g*end_state[1])-ro*Ap1*math.sqrt(2*g*end_state[0]), \
         ro*Ap3*math.sqrt(2*g*end_state[2])-ro*Ap2*math.sqrt(2*g*end_state[2])) 

# h1 [m]
x1_min, x1_max = initial_state[0]+0, initial_state[0]+.22

# h2 [m]
x2_min, x2_max = initial_state[1]+0, initial_state[1]+.22

# h3 [m]
x3_min, x3_max = initial_state[2]+0, initial_state[2]+.22

x11 = np.linspace(initial_state[0]+0, initial_state[0]+0.14, 8)
x12 = np.linspace(initial_state[0]+0.155, initial_state[0]+.220, 14)
x1_list = np.concatenate((x11 ,x12))
x21 = np.linspace(initial_state[1]+0, initial_state[1]+0.1, 11)
x22 = np.linspace(initial_state[1]+0.11, initial_state[1]+.220, 23)
x2_list = np.concatenate((x21 ,x22))
x31 = np.linspace(initial_state[2]+0, initial_state[2]+0.14, 8)
x32 = np.linspace(initial_state[2]+0.155, initial_state[2]+.220, 14)
x3_list = np.concatenate((x31 ,x32))

u11 = np.linspace(0, .1, 6)
u12 = np.linspace(.15, .6, 10)
u1_list = np.concatenate((u11, u12))
u21 = np.linspace(0, .2, 11)
u22 = np.linspace(.25, .6, 8)
u2_list = np.concatenate((u21, u22))
u31 = np.linspace(0, .1, 6)
u32 = np.linspace(.15, .6, 10)
u3_list = np.concatenate((u31, u32))

u1_list = [round(i,4) for i in u1_list]
u2_list = [round(i,4) for i in u2_list]
u3_list = [round(i,4) for i in u3_list]

# nonlinear discrete equations:
# x1(k+1) = x1(k) - dt*Ap1/A1*sqrt(2*g)*sqrt(x1(k)) + dt/(ro*A1)*u1(k)
# x2(k+1) = x2(k) + dt*Ap1/A2*sqrt(2*g)*sqrt(x1(k)) - dt*Ap2/A2*sqrt(2*g)*sqrt(x2(k)) + dt/(ro*A2)*u2(k)
# x3(k+1) = x3(k) + dt*Ap2/A3*sqrt(2*g)*sqrt(x1(k)) - dt*Ap3/A3*sqrt(2*g)*sqrt(x3(k)) + dt/(ro*A3)*u3(k)
# with coefficients
# x1(k+1) = c11*x1(k) + c12*sqrt(x1(k)) + c13*u1(k)
# x2(k+1) = c21*x2(k) + c22*sqrt(x1(k)) + c23*sqrt(x2(k)) + c24*u2(k)
# x3(k+1) = c31*x3(k) + c32*sqrt(x2(k)) + c33*sqrt(x3(k)) + c34*u3(k)

c11 =  1
c12 = -dt*Ap1/A1*math.sqrt(2*g)
c13 =  dt/(ro*A1)
c21 =  1
c22 =  dt*Ap1/A2*math.sqrt(2*g)
c23 = -dt*Ap2/A2*math.sqrt(2*g)
c24 =  dt/(ro*A2)
c31 =  1
c32 =  dt*Ap2/A3*math.sqrt(2*g)
c33 = -dt*Ap3/A3*math.sqrt(2*g)
c34 =  dt/(ro*A3)

print("c11: ", c11)
print("c12: ", c12)
print("c13: ", c13)
print("c21: ", c21)
print("c22: ", c22)
print("c23: ", c23)
print("c24: ", c24)
print("c31: ", c31)
print("c32: ", c32)
print("c33: ", c33)
print("c34: ", c34)

# linear continuous equation x_dot = Ac*x + Bc*u 
# around h10 = x10, h20 = x20, h30 = x30
x10 = .5
x20 = .6
x30 = .7
a11c = -Ap1/A1*g/math.sqrt(2*g*x10)
a12c = 0
a13c = 0
a21c = Ap1/A2*g/math.sqrt(2*g*x10)
a22c = -Ap2/A2*g/math.sqrt(2*g*x20)
a23c = 0
a31c = 0
a32c = Ap2/A3*g/math.sqrt(2*g*x20)
a33c = -Ap3/A3*g/math.sqrt(2*g*x30)
b11c = 1/(ro*A1)
b12c = 0
b13c = 0
b21c = 0
b22c = 1/(ro*A2)
b23c = 0
b31c = 0
b32c = 0
b33c = 1/(ro*A3)

# continuous matrices
Ac = np.matrix([[a11c,  a12c,  a13c],
                [a21c,  a22c,  a23c],
                [a31c,  a32c,  a33c]])

Bc = np.matrix([[b11c,   b12c,  b13c],
                [b21c,   b22c,  b23c],
                [b31c,   b32c,  b33c]])

# linear discrete x_k_1 = Ad*x_k + Bd*u_k
# around h10 = x10, h20 = x20, h30 = x30
a11d = 1+dt*a11c
a12d = 0
a13d = 0
a21d = dt*a21c
a22d = 1+dt*a22c
a23d = 0
a31d = 0
a32d = dt*a32c
a33d = 1+dt*a33c
b11d = dt*b11c
b12d = 0
b13d = 0
b21d = 0
b22d = dt*b22c
b23d = 0
b31d = 0
b32d = 0
b33d = dt*b33c

# matrices discrete
A = np.matrix([[a11d,   a12d,   a13d],
               [a21d,   a22d,   a23d],
               [a31d,   a32d,   a33d]])

B = np.matrix([[b11d,   b12d,   b13d],
               [b21d,   b22d,   b23d],
               [b31d,   b32d,   b33d]])

A_lqr, B_lqr = A, B

Q = np.matrix("1 0 0; 0 1 0; 0 0 1")
R = np.matrix(".1 0 0; 0 .1 0; 0 0 .1")

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
print("x3_list: ", x3_list)
print("u1_list: ", u1_list)
print("u2_list: ", u2_list)
print("u3_list: ", u3_list)

              
