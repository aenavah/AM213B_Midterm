import numpy as np
import matplotlib.pyplot as plt

import Midterm_functions
plot_boundary = Midterm_functions.plot_boundary 
RK3_AbsoluteStabilityRegion = Midterm_functions.RK3_AbsoluteStabilityRegion
beam_shooting_method = Midterm_functions.beam_shooting_method
beam_analytical_solution = Midterm_functions.beam_analytical_solution
Q1_plots = Midterm_functions.Q1_plots
RK3_delta_tstar = Midterm_functions.RK3_delta_tstar
Q2d_RK3 = Midterm_functions.Q2d_RK3
#-----------------------------------------------------
''' 
Question 1 Beam Problem
'''
N = 60000
dt = 1/N
T = 1
curv0 = np.array([5, 10]) #guess of curvature
tol = 10**(-6)

#analytical solution
analytical_data = beam_analytical_solution(T, dt)
#numerical solution
numerical_data = beam_shooting_method(dt, T, N, curv0, tol)

#plotting both on one figure
Q1_plots([analytical_data["Location"], numerical_data["Location"]], [analytical_data["Displacement"], numerical_data["Displacement"]], ["Analytical Solution", "Numerical Solution"], "Displacement", "Position", "Fully Clamped Euler-Bernoulli Beam", "Q1_Beam_Comparison.jpg")

#plotting beam error between analytical and numerical solution
beam_error = abs(analytical_data["Displacement"] - numerical_data["Displacement"])
Q1_plots([analytical_data["Location"]], [beam_error], ["Error"], "Position", "Error (log)", "Error between Analytical and Numerical Solution", "Q1d_Shooting_Error.jpg", "log")

#-----------------------------------------------------

'''
Question 3b : Plot the region of Stability
'''
def LMM_Boundary(theta):
  '''LMM method from Q3 boundary'''
  y = 4*(-1 - np.e**(1j*theta) - np.e**(2j*theta) + 3*np.e**(3j*theta))/(3 - 2*np.e**(1j*theta) + 23*np.e**(2j*theta))
  return y 

plot_boundary(LMM_Boundary)
plt.title("LMM (7) Absolute Stability Region")
plt.savefig("Q3b_LMM_AbsoluteStabilityRegion.jpg")
plt.clf()
#-----------------------------------------------------

'''
Question 2b : Plot the region of absolute stability of the RK3 method
'''
A = np.array([[0, 0, 0],
              [1/4, 1/4, 0],
              [0, 1, 0]])
b = np.array(([1/6],[2/3],[1/6]))
h = np.ones_like(b)
min_real = RK3_AbsoluteStabilityRegion(A, b, h)

'''
 Question 2c: Determine the largest delta t for which the RK3 applied to dy/dt = By
'''

B = np.array([[-1, 3, -5, 7],
             [0, -2, 4, -6],
             [0, 0, -4, 6],
             [0, 0, 0, -16]])

delta_tstar, typee = RK3_delta_tstar(B, min_real)
print("delta tstar: " + str(delta_tstar))

T = 15
dt = delta_tstar 
diff = -0.1
dt = delta_tstar + diff

n = int(T/dt)


ts = np.arange(0, T, dt)
ts = ts[0:n]

u = np.zeros((n , 4))
u[0] = np.array([1,1,1,1])
for i in range(1, n):
  u_next = Q2d_RK3(B, u[i-1], dt)
  u[i] = u_next
  u_last = u_next

plt.plot(ts, u[:,0], label = "dy1")  
plt.plot(ts, u[:,1], label = "dy2") 
plt.plot(ts, u[:,2], label = "dy3") 
plt.plot(ts, u[:,3], label = "dy4") 

plt.legend()
plt.grid()
plt.title(str(typee) +  "RK3 method with dt=dt* + "+str(diff))
print("type: " + str(typee))
plt.savefig("Q2d_dt=dtstar=" + str(delta_tstar) +"+" + str(diff) + "_" + typee + "_.jpg")
#plt.show()



