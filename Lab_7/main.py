import numpy as np

from random import randint

from functions import B_solve, matrix_norm, vector_norm, check_diagonal_dominance
from methods import simple_iteration_solve, jacobi_solve, seidel_solve

def main():

    # рандомна генерація матриці
    n = 100
    x_idial = 2.5
    A = np.random.uniform(1, 100 , (n, n))
    for i in range(n):
        A[i, i] += 5000.0

    check_diagonal_dominance(A)

    B_solve(A, x_idial, n)

    A = np.loadtxt("Lab_7/data/matrixA.txt", dtype=np.float64)
    B = np.loadtxt("Lab_7/data/vectorB.txt", dtype=np.float64)

    X = np.full(n, 1.0)

    # обчислення ітераційного параметра 
    tau = 1.0 / matrix_norm(n, A)

    eps = 1e-14
    max_iter = 1000

    res_si = simple_iteration_solve(n, A, B, X, tau, eps, max_iter)
    res_jac = jacobi_solve(n, A, B, X, eps, max_iter)
    res_sei = seidel_solve(n, A, B, X, eps, max_iter)

    print("\nМетод простої ітерації:", vector_norm(n, res_si, x_idial))
    print("Метод Якобі:", vector_norm(n, res_jac, x_idial))
    print("Метод Зейделя:", vector_norm(n, res_sei, x_idial))

if __name__ == "__main__":
    main()