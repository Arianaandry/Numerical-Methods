import numpy as np

from LU import solve_SLAE, LU_error, A_X_product


def clarification_SLAE(X_calc, n, A, B, L, U):
    eps = 1e-14
    X = X_calc.copy()
    iter = 0
    previous_error = float('inf')

    while iter < 100:
        AX = A_X_product(n, A, X)
        
        R = np.zeros(n)
        for i in range(n):
            R[i] = B[i] - AX[i]

        cur_error = LU_error(n, A, X, B)
        print(f"Ітерація {iter}: Похибка = {cur_error}")

        np.savetxt("Lab_6/files/iterations.txt", X, fmt="%.18f")

        if cur_error <= eps or cur_error >= previous_error:
            if cur_error >= previous_error and iter > 0:
                print("Похибка перестала зменшуватися.")
            break

        previous_error = cur_error

        delta_X = solve_SLAE(n, R, L, U)
        
        for i in range(n):
            X[i] = X[i] + delta_X[i]
            
        iter += 1

   