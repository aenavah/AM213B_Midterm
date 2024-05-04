import matplotlib.pyplot as plt 
import numpy as np 




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