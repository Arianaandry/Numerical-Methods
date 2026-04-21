from matplotlib import pyplot as plt


def show_server_load_function(x, y):
    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')

    plt.title('Графік функції навантаження на сервер')
    plt.xlabel('Час(год), x')
    plt.ylabel('Навантаження, f(x)')

    plt.grid(True)
    plt.legend()
    plt.show()


def show_N_error_dependence(N_values, errors, n_opt, eps_opt, target_eps=1e-12):
    plt.figure(figsize=(10, 6))

    plt.semilogy(N_values, errors, color='darkgreen', linewidth=1.2 , label='|I(N) - I0|')
    plt.axhline(y=target_eps, color='r', linestyle='--', linewidth=1, label='Задана точність 1e-12')
    if n_opt:
        plt.plot(n_opt, eps_opt, 'ro') # Точка оптимального N
        
    plt.title('Залежність похибки методу Сімпсона від числа розбиттів N')
    plt.xlabel('Число розбиттів N')
    plt.ylabel('Абсолютна похибка (log scale)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.show()