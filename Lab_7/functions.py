import numpy as np

    # розрахунок  В
def B_solve(A, xi, n) -> None:
    B = np.zeros(n) 

    for i in range(n):
        B[i] = sum(A[i, j] * xi for j in range(n))

    np.savetxt("Lab_7/data/matrixA.txt", A, fmt="%.18f")
    np.savetxt("Lab_7/data/vectorB.txt", B, fmt="%.18f")

    # добуток матриці на вектор
def A_X_product(n, A, X):
    AX = np.zeros(n)
    for i in range(n):
        AX[i] = sum(A[i, j] * X[j] for j in range(n))

    return AX

# норма вектора
def vector_norm(n, X, x_ideal=2.5):
    max_val = 0.0
    for i in range(n):
        diff = abs(X[i] - x_ideal)
        if diff > max_val:
            max_val = diff
    
    return max_val

# Обчислення норми матриці (максимальна сума модулів елементів рядків)
def matrix_norm(n, C):
    max_sum = 0.0
    for i in range(n):

        current_row_sum = sum(abs(C[i, j]) for j in range(n))
        
        if current_row_sum > max_sum:
            max_sum = current_row_sum
            
    return max_sum

def check_diagonal_dominance(A):
    n = len(A)
    sums = np.zeros(n)

    for i in range(n):
        sums[i] = sum(abs(A[i, j]) for j in range(n) if j != i)

    np.savetxt("Lab_7/data/check_diagonal_dominance.txt", sums, fmt="%.18f")
    return sums