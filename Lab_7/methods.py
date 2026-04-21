import numpy as np

from functions import A_X_product

# метод простої ітерації
def simple_iteration_solve(n, A, f, X, tau, eps, max_iter):
    current_X = X.copy()
    for k in range(max_iter):

        AX = A_X_product(n, A, current_X)
        
        X_new = np.zeros(n)
        for i in range(n):
            # x_i(k+1) = x_i(k) - tau * ( (AX)_i - f_i )
            X_new[i] = current_X[i] - tau * (AX[i] - f[i])
        
        max_diff = max(abs(X_new[i] - current_X[i]) for i in range(n))
        current_X = X_new

        if max_diff < eps:
            print(f"Метод простої ітерації збігся за {k+1} ітерацій")
            np.savetxt("Lab_7/data/simple_iteration.txt", current_X, fmt="%.18f")
            return current_X
    print("Метод простої ітерації не збігся за задану кількість ітерацій")
    return current_X

# метод Якобі
def jacobi_solve(n, A, f, X, eps, max_iter):
    current_X = X.copy() # Початкове наближення (у нашому випадку всі 1.0)
    for k in range(max_iter):
        X_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * current_X[j] for j in range(n) if i != j)
            X_new[i] = (f[i] - s) / A[i, i]
        
        if max(abs(X_new[i] - current_X[i]) for i in range(n)) < eps:
            print(f"Метод Якобі збігся за {k+1} ітерацій")
            np.savetxt("Lab_7/data/jacobi.txt", X_new, fmt="%.18f")
            return X_new
        current_X = X_new
    print("Метод Якобі не збігся за задану кількість ітерацій")
    return current_X

# метод Зейделя
def seidel_solve(n, A, f, X, eps, max_iter):
    current_X = X.copy()
    for k in range(max_iter):
        X_old = current_X.copy()
        for i in range(n):
            s_new = sum(A[i, j] * current_X[j] for j in range(i))
            s_old = sum(A[i, j] * X_old[j] for j in range(i + 1, n))
            current_X[i] = (f[i] - s_new - s_old) / A[i, i]
        
        if max(abs(current_X[i] - X_old[i]) for i in range(n)) < eps:
            print(f"Метод Зейделя збігся за {k+1} ітерацій")
            np.savetxt("Lab_7/data/seidel.txt", current_X, fmt="%.18f")
            return current_X    
    print("Метод Зейделя не збігся за задану кількість ітерацій")
    return current_X