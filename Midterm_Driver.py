import numpy as np
import matplotlib.pyplot as plt

import Midterm_functions
plot_boundary = Midterm_functions.plot_boundary 
RK3_AbsoluteStabilityRegion = Midterm_functions.RK3_AbsoluteStabilityRegion

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

'''
Question 2b : Plot the region of absolute stability of the RK3 method
'''
A = np.array([[0, 0, 0],
              [1/4, 1/4, 0],
              [0, 1, 0]])
b = np.array(([1/6],[2/3],[1/6]))
h = np.ones_like(b)


RK3_AbsoluteStabilityRegion(A, b, h)

'''
 Question 2c: Determine the largest delta t for which the RK3 applied to dy/dt = By
'''
B = np.array([-1, 3, -5, 7],
             [0, -2, 4, -6],
             [0, 0, -4, 6],
             [0, 0, 0, -16])

