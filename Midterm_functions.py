import matplotlib.pyplot as plt 
import numpy as np 
from math import atan2



def plot_boundary(method):
  '''takes in a functions "method" that takes in theta and ouputs y at theta'''
  #plots boundary given the boundary function
  # note: need to plt.title and plt.show after calling this function
  plt.clf()
  thetas = np.linspace(0, 2*np.pi)

  real_vals = []
  imag_vals = []
  
  #plotting actual boundary
  for theta in thetas:
    y = method(theta) 
    real_part = y.real
    imag_part = y.imag
    real_vals.append(real_part)
    imag_vals.append(imag_part)
  plt.grid()
  plt.xlabel("Re(z)")
  plt.ylabel("Im(z)")
  plt.plot(real_vals, imag_vals)  

def RK3_S(A, b, h, x, y):
  z = complex(x,y)
  S = abs(np.linalg.det(np.eye(3) - (z*A) + z*(h@(b.T))))
  return S

def RK3_AbsoluteStabilityRegion(A, b, h):
  nx, ny = (200, 200)
  imag = np.linspace(-4, 4, nx)
  real = np.linspace(-6, 2, ny)
  real_v, imag_v = np.meshgrid(real, imag)

  S_grid = np.ones((nx, ny))

  for x_i in range(nx):
    for y_i in range(ny):
      boundary = RK3_S(A, b, h, real_v[x_i, y_i], imag_v[x_i, y_i])
      S_grid[x_i, y_i] = boundary 

  plt.xlabel("Re(z)")
  plt.ylabel("Im(z)")
  plt.contour(real_v, imag_v, S_grid, [1])
  plt.grid()
  plt.title("Boundary of RK3 Method")
  plt.savefig("Q2b_RK3_Boundary.jpg")

#Question 1b
def RK4(x, y0, h, f):
  K1 = f(x, y0)
  K2 = f(x + (1/2) * h, y0 + ((1/2)*h * K1))
  K3 = f(x + (1/2) * h, y0 + ((1/2)*h * K2))
  K4 = f(x + (1) * h, y0 + h*K3)

  y_next = y0 + (h * (1/6)) * (K1 + 2*K2 + 2*K3 + K4)
  return y_next  

def f(x, z):
    q = x**2
    E = 1
    I = 1
    array = np.array([z[1], z[2], z[3], q/(E*I)])
    return array 

def f_eta(x, y):
  J_eta = np.array(
                  [[0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])
  eta_next = J_eta @ y 
  return eta_next

def shooting_method(dt, T, n, v0, tol):
  ts = np.arange(0, T, dt)
  ts = ts[0 : n]
  E = np.array([1, 1])
  z0 = np.array([0, 0, v0[0], v0[1]])
  v = v0
  z = z0
  while np.linalg.norm(E) > tol:
    plt.clf()  
    x = []
    y = []

    
    z = np.array([0, 0, v[0], v[1]])
    for i in range(1, int(n)):
      z_next = RK4(ts[i], z, dt, f)
      z = z_next
      x.append(ts[i])
      y.append(z[0])

    #step 3 
    eta_0 = np.array([0, 0, 0, 0, 1, 0, 0, 1])

    eta = eta_0
    for i in range(1, int(n)):
      eta_next = RK4(ts[i], eta, dt, f_eta)
      eta = eta_next

    #step 4
    #print(eta)
    J = np.array([[eta[0], eta[1]],
                [eta[2], eta[3]]])
    
    #step 5 
    E = np.array([z[0], z[1]])
    v_next = v - np.linalg.inv(J)@E
    v = v_next
    
  plt.title("Fully Clamped Euler-Bernoulli Beam")
  plt.plot(x, y, color = "pink")
  plt.ylabel("Displacement")
  plt.xlabel("Position")
  plt.savefig("Q1b_Beam.jpg")