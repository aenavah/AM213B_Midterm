import matplotlib.pyplot as plt 
import numpy as np 
from math import atan2
import pandas as pd

#-----------------------------------------------------
#Question 1
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

def Q1_plots(xs, ys, labels, x_label, y_label,  title, figure, yscale = "linear"):
  plt.clf()
  for i in range(len(xs)):
    plt.plot(xs[i], ys[i], label = labels[i])
  plt.legend()
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.yscale(yscale)
  plt.title(title)
  plt.grid()
  plt.savefig(figure)

def beam_shooting_method(dt, T, n, v0, tol):
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

    eta_0 = np.array([0, 0, 0, 0, 1, 0, 0, 1])
    eta = eta_0
    for i in range(1, int(n)):
      eta_next = RK4(ts[i], eta, dt, f_eta)
      eta = eta_next

    J = np.array([[eta[0], eta[1]],
                [eta[2], eta[3]]])
    
    E = np.array([z[0], z[1]])
    v_next = v - np.linalg.inv(J)@E
    v = v_next

  print(v)
  #plot  
  Q1_plots([x], [y], ["Numerical Solution"], "Position", "Displacement",  "Fully Clamped Euler-Bernoulli Beam", "Q1b_Beam_Numerical.jpg")
  plt.clf()

  #data
  data = {"Location": x, 
          "Displacement": y}
  df = pd.DataFrame(data)
  return df

def beam_analytical_solution(T, dt):
  def bridge_analytical(x):
    f = (1/360)*x**6 - (1/90)*x**3 + (1/120)*x**2
    return f 

  analytical_linspace = np.arange(0,T, dt)
  sols = bridge_analytical(analytical_linspace)

  #plot
  Q1_plots([analytical_linspace], [sols], ["Analytical Solution"], "Position", "Displacement",  "Fully Clamped Euler-Bernoulli Beam", "Q1a_Beam_Analytical.jpg")
  plt.clf()

  #data storing 
  data = {"Location": analytical_linspace, 
          "Displacement": sols}
  df = pd.DataFrame(data)
  return df

#-----------------------------------------------------
#Question 3
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
#-----------------------------------------------------  
# Question 2
def RK3_S(A, b, h, x, y):
  z = complex(x,y)
  S = abs(np.linalg.det(np.eye(3) - (z*A) + z*(h@(b.T))))
  return S

def RK3_AbsoluteStabilityRegion(A, b, h):
  nx, ny = (200+1, 200+1)
  imag = np.linspace(-4, 4, nx)
  real = np.linspace(-6, 2, ny)
  real_v, imag_v = np.meshgrid(real, imag)

  S_grid = np.ones((nx, ny))

  #iterate through y values for each fixed x
  for x_i in range(nx):
    for y_i in range(ny):
      boundary = RK3_S(A, b, h, real_v[x_i, y_i], imag_v[x_i, y_i])
      S_grid[x_i, y_i] = boundary 
      #print(S_grid)
  plt.xlabel("Re(z)")
  plt.ylabel("Im(z)")
  contour = plt.contourf(real_v, imag_v, S_grid)
  plt.colorbar()
  plt.grid()
  plt.title("Contours of RK3 Method")
  plt.savefig("Q2b_RK3_Contours.jpg")
  #plt.show()

  # Extract x, y values for the contour at height 1
  contour_x = []
  contour_y = []
  for line in contour.collections[0].get_paths():
    vertices = line.vertices
    contour_x.extend(vertices[:, 0])
    contour_y.extend(vertices[:, 1])

  plt.clf()
  plt.title("Boundary of RK3 Method")
  #plt.savefig("Q2b_RK3_Boundary.jpg")
  
  plt.ylim(-4, 4)
  plt.xlim(-6, 2)
  
  plt.xlabel("Re(z)")
  plt.ylabel("Im(z)")
  plt.plot(contour_x, contour_y)
  #print(min(contour_x))
  min_real = min(contour_x)
  return min_real 

#Q2c
def RK3_delta_tstar(B, min_real):
  eigenvalues = np.linalg.eigvals(B)
  for dt in np.arange(1, 0, -.000001):
    need_toscale = min(eigenvalues)
    scaled_eig = dt*need_toscale
    if abs(scaled_eig) < abs(min_real):
      scaled_eigens = []
      dtstar = dt
      for eig in eigenvalues:
        new_eigs = dt*eig
        scaled_eigens.append(new_eigs)
      plt.grid()
      plt.scatter(scaled_eigens, [0,0,0,0], s = 40, facecolors = "none", edgecolors = "r", label = "Eigenvalues")
      plt.savefig("Q2b_ScaledEigenvalues.jpg")
      plt.clf()
      return dtstar
      break

def Q2d_RK3_f(B, x):
  return B @ x

def Q2d_RK3(A, u0, dx):
    f = Q2d_RK3_f 
    '''RK method from HW1'''
    K1 = f(A, u0)
    K2 = f(A + (1/2) * dx , u0 + ((1/4) * dx * K1) + ((1/4) * dx * K2))
    K3 = f(A + dx, u0 + dx * K2)
    u_next = u0 + (dx * (1/6)) * (K1 + 4 * K2 + K3)
    return u_next
