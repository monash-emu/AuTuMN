"""
This module is to fit a logistic function to mutiple coverage-cost data points
tdoan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import math

def logistic(p, x):
    x0, y0, c, k = p
    y = y0 - k * np.log (( c - x + x0) / (x + x0))

    return y
    #y = spending
    # x = coverage
    #c = difference between lower (y0) and upper (saturation point) asymptotes
    #k = steepness of the curve
    #x0 = reflection (mid) point
    #y0 = lower symptote

def residuals(p, x, y):
    return y - logistic(p, x)

# raw data
y = np.array([821, 576, 473, 377, 326], dtype='float')
x = np.array([255, 235, 208, 166, 157], dtype='float')


print(np.median(x))
print(np.median(y))
#print(x)
#print(y)
p_guess = (np.median(x), np.median(y), 100, 100)
p, cov, infodict, mesg, ier = scipy.optimize.leastsq(residuals, p_guess, args=(x, y),full_output=1)

x0, y0, c, k = p

print('''\
x0 reflection point = {x0}
y0 lower asymptote = {y0}
c difference between upper and lower asymptote= {c}
k steepness of curve = {k}
'''.format(x0 = x0, y0 = y0, c = c, k = k))

xp = np.linspace(100, np.max(x) + 0.8 * np.max(x), 500)

pxp = logistic(p, xp)

# Plot the results
#fig = plt.figure(1)
#plt.plot(x, y, 'o', xp, pxp, 'r-', linewidth = 2)
#plt.xlabel('% Coverage')
#plt.ylabel('$ Spending',rotation='vertical')
#plt.grid(True)
#fig.suptitle('Cost-coverage curve')
#plt.show()