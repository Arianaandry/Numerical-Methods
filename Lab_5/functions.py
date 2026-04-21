import numpy as np
from scipy.integrate import quad


calls = 0
def f(x):
    global calls
    calls += len(np.atleast_1d(x))
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

# наближене значення означеного інтегралу по N
def simpson_method(func, a, b, N):

    if N % 2 != 0:
        raise ValueError("N must be an even number.")
    
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = func(x)

    integral = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])

    return (h / 3) * integral, x, y

# пошук значень, де похибка > 10^-12    Зав.4
def N_error_dependence(a, b):
    I0, _ = quad(f, 0, 24)            # зав.2
    N_values = np.arange(10, 1002, 2)
    errors = []
    target_eps = 1e-12
    n_opt = None
    eps_opt = None

    for N_val in N_values:
        I_approx, _, _ = simpson_method(f, a, b, N_val)
        err = abs((I_approx - I0))
        errors.append(err)

        if n_opt is None and err <= target_eps:
            n_opt = N_val
            eps_opt = err

    return I0, N_values, errors, n_opt, eps_opt


def adaptive_simpson(f, a, b, eps, whole_integral=None):
 
    c = (a + b) / 2
    h = b - a
    
   
    f_a, f_b, f_c = f(a), f(b), f(c)
    f_left_mid = f((a + c) / 2)
    f_right_mid = f((c + b) / 2)
    
  
    if whole_integral is None:
        whole_integral = (h / 6) * (f_a + 4 * f_c + f_b)
        
  
    left_S = (h / 12) * (f_a + 4 * f_left_mid + f_c)
    right_S = (h / 12) * (f_c + 4 * f_right_mid + f_b)
    
   
    if abs(left_S + right_S - whole_integral) <= 15 * eps:
      
        return left_S + right_S + (left_S + right_S - whole_integral) / 15
    
   
    return (adaptive_simpson(f, a, c, eps / 2, left_S) + 
            adaptive_simpson(f, c, b, eps / 2, right_S))