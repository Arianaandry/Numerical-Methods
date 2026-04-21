import numpy as np

from random import randint

from LU import B_solve, L_U_solve, LU_error, solve_SLAE
from refinement import i_refinement_SLAE

def main():

    # рандомна генерація матриці
    n = 100
    x_idial = 2.5
    A = np.random.uniform(1, 100 , (n, n))
    for i in range(n):
        A[i, i] += 5000.0
    X = np.full(n, x_idial)

    B_solve(A, x_idial, n)
    L_U_solve(n, A)

    A = np.loadtxt("Lab_6/files/matrixA.txt", dtype=np.float64)
    B = np.loadtxt("Lab_6/files/vectorB.txt", dtype=np.float64)
    L = np.loadtxt("Lab_6/files/matrixL.txt", dtype=np.float64)
    U = np.loadtxt("Lab_6/files/matrixU.txt", dtype=np.float64)

    X_calc = solve_SLAE(n, B, L, U)

    error = LU_error(n, A, X_calc, B)
    print(f"\nПохибка LU розкладу: {error:.18f}")

    print()
    i_refinement_SLAE(X_calc, n, A, B, L, U)

if __name__ == "__main__":
    main()