import sys
import numpy as np
from math import factorial, pow, floor, ceil, log10
import matplotlib.pyplot as plt
from scipy.stats import binom, gamma
from scipy.special import comb
from scipy.interpolate import make_interp_spline

# Q1
# Get probability within one error from expecation with input n, p, e
def run_case(n, p, e):
    expecation = n * p
    lower_bound = ceil(expecation * (1 - e))
    upper_bound = floor(expecation * (1 + e))
    result = 0
    for i in range(lower_bound, upper_bound + 1):
        result = result + binom.pmf(i, n, p)
    return result

# Print result
print(' --------------------------------------------')
print('|  n \\ p   | p=0.25, e=0.01 | p=0.05, e=0.05 |')
print(' --------------------------------------------')
print('|  n=100   |  %.4f        |  %.4f        |' % (run_case(100, 0.25, 0.01), run_case(100, 0.05, 0.05)))
print(' --------------------------------------------')
print('|  n=3000  |  %.4f        |  %.4f        |' % (run_case(3000, 0.25, 0.01), run_case(3000, 0.05, 0.05)))
print(' --------------------------------------------')
print('|  n=50000 |  %.4f        |  %.4f        |' % (run_case(50000, 0.25, 0.01), run_case(50000, 0.05, 0.05)))
print(' --------------------------------------------')

# Q2
# Compute all in log10
comb1 = [log10(comb(5, k)) for k in range(0, 5 + 1)]
comb2 = [log10(comb(50, k)) for k in range(0, 50 + 1)]
prob1 = []
prob2 = []
binomial1 = []
binomial2 = []
for i in range(len(comb1)):
    prob1.append(log10(binom.pmf(i, 5, 0.1)) - comb1[i])
    binomial1.append(comb1[i] + prob1[i])
for i in range(len(comb2)):
    prob2.append(log10(binom.pmf(i, 50, 0.1)) - comb2[i])
    binomial2.append(comb2[i] + prob2[i])

# Smoothing by transfer into 300 points
x1 = [(1 / (len(comb1) - 1)) * i for i in range(len(comb1))]
x2 = [(1 / (len(comb2) - 1)) * i for i in range(len(comb2))]
x = np.linspace(0, 1, 300)
comb1 = make_interp_spline(x1, comb1)(x)
comb2 = make_interp_spline(x2, comb2)(x)
prob1 = make_interp_spline(x1, prob1)(x)
prob2 = make_interp_spline(x2, prob2)(x)
binomial1 = make_interp_spline(x1, binomial1)(x)
binomial2 = make_interp_spline(x2, binomial2)(x)

# Plot
plt.plot(x, binomial1, label='binomial_n=5_p=0.1', color="blue")
plt.plot(x, comb1, label='combination_n=5_p=0.1', color="orange")
plt.plot(x, prob1, label='probability_n=5_p=0.1', color="green")
plt.plot(x, binomial2, label='binomial_n=50_p=0.1', color="red")
plt.plot(x, comb2, label='combination_n=50_p=0.1', color="purple")
plt.plot(x, prob2, label='probability_n=50_p=0.1', color="brown")
plt.legend(loc='upper right')
plt.show()