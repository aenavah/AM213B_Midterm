import numpy as np
import matplotlib.pyplot as plt

import Midterm_functions
plot_boundary = Midterm_functions.plot_boundary 


### Question 3b : Plot the region of Stability
def LMM_Boundary(theta):
  '''LMM method from Q3 boundary'''
  y = 4*(-1 - np.e**(1j*theta) - np.e**(2j*theta) + 3*np.e**(3j*theta))/(3 - 2*np.e**(1j*theta) + 23*np.e**(2j*theta))
  return y 

plot_boundary(LMM_Boundary)
plt.title("LMM (7) Absolute Stability Region")
plt.savefig("Q3b_LMM_AbsoluteStabilityRegion.jpg")
