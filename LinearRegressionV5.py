import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import ctypes

X = np.array([4, 3, 1, 0, 6, 2, 7, 8, 5, 0, 2, 1, 0, 1, 2, 3]) #[10, 20, 30]  [7, 14, 21] [10, 20, 30] [100, 200, 300]
Y = np.array([5, 3, 0, 0, 7, 1, 9, 7, 7, 0, 1, 1, 1, 0, 0, 1]) #[5, 10, 15] [5, 10, 15] [1, 2, 3] [1, 2, 3]
PARAMETERS = [2, 1]
LEARNING_RATE = 0.001 #ITS THE LEARNING RATE 0.00001 = X in 100s , 0.0000001 = X in 1000s

def h(x, parameters):
    return parameters[0] + (parameters[1] * x)

def costFunction(hypothesis, y):
    COST_FUNCTION_SUM = np.sum(np.square(np.subtract(hypothesis, y)))
    COST_FUNCTION_AVERAGE = (1/(2*len(X))) * COST_FUNCTION_SUM
    return COST_FUNCTION_AVERAGE

def derivative(parameters, hypothesis, y, x):
    NEW_PARAMETER0 = parameters[0] - LEARNING_RATE * ((1/len(hypothesis)) * sum(hypothesis - y))
    NEW_PARAMETER1 = parameters[1] - LEARNING_RATE * (((1/len(hypothesis)) * sum((hypothesis - y) * x)))
    return NEW_PARAMETER0, NEW_PARAMETER1

for i in range(100_000):
    H_RESULTS = h(X, PARAMETERS)
    COST_FUNCTION_AVERAGE = costFunction(H_RESULTS, Y)
    print(COST_FUNCTION_AVERAGE)
    P0, P1 = derivative(PARAMETERS, H_RESULTS, Y, X)
    PARAMETERS[0] = P0
    PARAMETERS[1] = P1