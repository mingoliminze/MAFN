# This is the Python Code of Homework 2 
# @Author: Minze Li
# @Date: Sep 26, 2024
# MATH GR5430 Machine Learning for Finance
import numpy as np
from sklearn.linear_model import Ridge

# (a) Compute the Moore-Penrose pseudoinverse of a matrix using SVD.
def pseudoInverse(A):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.where(s > np.finfo(float).eps, 1/s, 0)
    return Vt.T @ np.diag(s_inv) @ U.T

# (b) Test the pseudoinverse function by generating a random invertible square matrix.
def invertibleTest(N=1000):
    success_cnt = 0
    for i in range(N):
        n = np.random.randint(2, 50)  # Random size between 2 and 50
        A = np.random.rand(n, n)
        if np.allclose(np.linalg.inv(A), pseudoInverse(A), atol=np.finfo(float).eps): # use the default numerical accuracy of NumPy
            success_cnt += 1
        else:
            print(f"Test {i+1} failed")
    
    print(f"Invertible matrix tests passed: {success_cnt}/{N}")
    print(f"Invertible matrix tests success rate: {success_cnt/N*100:.2f}%")

# (c) Test the pseudoinverse function against the limit of ridge regression
def ridgeTest(N=1000):
    success_cnt = 0
    for i in range(N):
        m, n = np.random.randint(5, 10, size=2)
        A = np.random.rand(m, n)
        # A = np.random.rand(n, n) if use sklearn.ridge
        A[:, -1] = A[:, 0] # Make last column linearly dependent
        
        A_psd_inv = pseudoInverse(A)
        #ridge = Ridge(alpha=1e-8, fit_intercept=False)
        #ridge.fit(A, np.eye(n))
        #ridge_coef = ridge.coef_.T
        lambda_reg = 1e-8
        ridge_inv = np.linalg.inv(A.T @ A + lambda_reg * np.eye(n)) @ A.T
        # if np.allclose(A_psd_inv, ridge_coef, atol=1e-4, rtol=1e-4):
        if np.allclose(A_psd_inv, ridge_inv, atol=1e-4, rtol=1e-4): # verify the approximate equality
            success_cnt += 1
        else:
            print(f"Test {i+1} failed")
    
    print(f"Ridge regression tests passed: {success_cnt}/{N}")
    print(f"Ridge regression tests success rate: {success_cnt/N*100:.2f}%")

# Run the tests
if __name__ == '__main__':
    invertibleTest()
    ridgeTest()