import numpy as np

def sinusoid(x):
    return list(np.sin(x))

def tangent(x):
    return list(np.tan(x))

def cosine(x):
    return list(np.cos(x))

def linear(x):
    return x + np.random.normal(0, 1)