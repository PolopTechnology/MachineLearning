import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

X = np.array([0.1, 0.2, 0.3]) #Input values  idk
Y = np.array([1, 2, 3]) #Expected output values  idk
PARAMETERS = np.array([1, 0.5])
LEARNING_RATE = 0.1

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

for i in range(500):
    H_RESULTS = h(X, PARAMETERS)
    COST_FUNCTION_AVERAGE = costFunction(H_RESULTS, Y)
    print(COST_FUNCTION_AVERAGE)
    P0, P1 = derivative(PARAMETERS, H_RESULTS, Y, X)
    PARAMETERS[0] = P0
    PARAMETERS[1] = P1