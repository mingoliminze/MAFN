# This is the Python Code of Homework 1 
# Author: Minze Li
# Date: Sep 8, 2024
# MATH GR5430 Machine Learning for Finance
import numpy as np

def bmatrix(a): # To output a matrix in LaTeX format
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

rng = np.random.default_rng() # Create a random number generator
a = rng.normal(size=(10, 5)) # Create a random 10x5 matrix
print(bmatrix(np.round(a, 4)))

q,r = np.linalg.qr(a)
print(bmatrix(np.round(q, 4)), bmatrix(np.round(r, 4)))
print(np.allclose(a, np.dot(q, r))) # Test if QR = A

q2, r2 = np.linalg.qr(a, mode='complete')
print(bmatrix(np.round(q2, 4)), bmatrix(np.round(r2, 4)))
print(np.allclose(a, np.dot(q2, r2)))

r3 = np.linalg.qr(a, mode='r')
print(np.allclose(r, r3)) # Test if R = R3

h, tau = np.linalg.qr(a, mode='raw')
print(bmatrix(np.round(h, 4)), np.round(tau, 4))
