import subprocess
import sys


# Перевірка та автоматичне встановлення бібліотек, якщо вони відсутні
def install_libraries():
    required = {'numpy', 'matplotlib', 'scipy'}
    try:
        import numpy, matplotlib, scipy
    except ImportError:
        print("[СИСТЕМА] Встановлення відсутніх бібліотек...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib", "scipy"])
        print("[СИСТЕМА] Бібліотеки встановлено. Перезапустіть скрипт.")
        sys.exit(0)

install_libraries()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

# Вхідні дані
data = {"results":[
  {"latitude":48.164214,"longitude":24.536044,"elevation":1264.0},
  {"latitude":48.164983,"longitude":24.534836,"elevation":1285.0},
  {"latitude":48.165605,"longitude":24.534068,"elevation":1285.0},
  {"latitude":48.166228,"longitude":24.532915,"elevation":1333.0},
  {"latitude":48.166777,"longitude":24.531927,"elevation":1310.0},
  {"latitude":48.167326,"longitude":24.530884,"elevation":1318.0},
  {"latitude":48.167011,"longitude":24.530061,"elevation":1318.0},
  {"latitude":48.166053,"longitude":24.528039,"elevation":1339.0},
  {"latitude":48.166655,"longitude":24.526064,"elevation":1375.0},
  {"latitude":48.166497,"longitude":24.523574,"elevation":1417.0},
  {"latitude":48.166128,"longitude":24.520214,"elevation":1486.0},
  {"latitude":48.165416,"longitude":24.517170,"elevation":1524.0},
  {"latitude":48.164546,"longitude":24.514640,"elevation":1553.0},
  {"latitude":48.163412,"longitude":24.512980,"elevation":1630.0},
  {"latitude":48.162331,"longitude":24.511715,"elevation":1757.0},
  {"latitude":48.162015,"longitude":24.509462,"elevation":1794.0},
  {"latitude":48.162147,"longitude":24.506932,"elevation":1828.0},
  {"latitude":48.161751,"longitude":24.504244,"elevation":1887.0},
  {"latitude":48.161197,"longitude":24.501793,"elevation":1975.0},
  {"latitude":48.160580,"longitude":24.500537,"elevation":1975.0},
  {"latitude":48.160250,"longitude":24.500106,"elevation":2031.0}
]}

results = data["results"]
n = len(results)

def haversine(lat1, lon1, lat2, lon2):
    """
    Обчислення відстані між двома точками на сфері (Гаверсинус) в метрах
    """
    R = 6371000 # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Вилучення даних
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])

# Обчислення кумулятивної відстані
distances = [0.0]
for i in range(1, n):
    d = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
    distances.append(distances[-1] + d)

distances = np.array(distances)

# --- 1. ВИВІД У КОНСОЛЬ (Табуляція) ---

print(f"Кількість точок: {n}")

print("\nТабуляція вузлів:")
print(" № | Latitude  | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

print("\nТабуляція (Кумулятивна відстань, Висота):")
print(" № | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:12.2f} | {elevations[i]:10.2f}")

# --- ДОДАТКОВИЙ РОЗРАХУНОК КОЕФІЦІЄНТІВ (МЕТОД ПРОГОНКИ) ---

def solve_tridiagonal(a, b, c, d):
    n_sys = len(d)
    cp = np.zeros(n_sys-1)
    dp = np.zeros(n_sys)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n_sys-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
    for i in range(1, n_sys):
        denom = b[i] - a[i-1] * cp[i-1]
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    x = np.zeros(n_sys)
    x[-1] = dp[-1]
    for i in range(n_sys-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def calculate_manual_coefficients(x, y):
    n_int = len(x) - 1
    h = np.diff(x)
    # СЛАР для c_i
    alpha = h[1:-1]
    beta = 2 * (h[:-1] + h[1:])
    gamma = h[1:-1]
    rhs = 3 * ((y[2:] - y[1:-1])/h[1:] - (y[1:-1] - y[:-2])/h[:-1])
    
    c_mid = solve_tridiagonal(alpha, beta, gamma, rhs)
    c = np.zeros(n_int + 1)
    c[1:-1] = c_mid
    
    a = y[:-1]
    b = (y[1:] - y[:-1])/h - h*(c[1:] + 2*c[:-1])/3
    d = (c[1:] - c[:-1])/(3*h)
    return a, b, c[:-1], d

# Розрахунок коефіцієнтів для максимальної кількості вузлів (20)
a_coeffs, b_coeffs, c_coeffs, d_coeffs = calculate_manual_coefficients(distances, elevations)

print("\nКоефіцієнти кубічних сплайнів:")
print(f"{'i':>3} | {'a_i':>10} | {'b_i':>10} | {'c_i':>10} | {'d_i':>10}")
print("-" * 55)
for i in range(len(a_coeffs)):
    print(f"{i:3d} | {a_coeffs[i]:10.2f} | {b_coeffs[i]:10.4f} | {c_coeffs[i]:10.4f} | {d_coeffs[i]:10.7f}")

# --- НОВІ РОЗДІЛИ (ХАРАКТЕРИСТИКИ МАРШРУТУ ТА ЕНЕРГІЯ) ---

print("\n Характеристики маршруту:")
print("Загальна довжина маршруту (м):", distances[-1])

total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
print("Сумарний набір висоти (м):", total_ascent)

total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))
print("Сумарний спуск (м):", total_descent)

# Аналіз градієнта (через похідну сплайна)
xx = np.linspace(distances.min(), distances.max(), 1000)
spline_full = make_interp_spline(distances, elevations, k=3)
yy_full = spline_full(xx)
grad_full = np.gradient(yy_full, xx) * 100 # у відсотках

print("\n Аналіз градієнта:")
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))

# Механічна енергія підйому
print("\n Механічна енергія підйому:")
mass = 80
g = 9.81
energy = mass * g * total_ascent
print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy / 1000)

# --- 2. ЗАПИС У ФАЙЛ ---
file_name = "tabulation_results.txt"
try:
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"Кількість вузлів: {n}\n")
        f.write("№ | Latitude | Longitude | Elevation (m) | Cum. Distance (m)\n")
        for i in range(n):
            p = results[i]
            f.write(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f} | {distances[i]:.2f}\n")
    print(f"\nТабуляцію збережено: {os.path.abspath(file_name)}")
except Exception as e:
    print(f"\nНе вдалося зберегти файл: {e}")

# --- 3. ПОБУДОВА ГРАФІКІВ ---
dist_smooth = np.linspace(distances.min(), distances.max(), 300)
knot_counts = [10, 15, 20]

plt.figure(figsize=(12, 10))

# Графік 1: Сплайни
plt.subplot(2, 1, 1)
plt.scatter(distances, elevations, color='black', label='Вихідні вузли', zorder=5)

for k in knot_counts:
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    k_dist = distances[idx]
    k_elev = elevations[idx]
    
    spline = make_interp_spline(k_dist, k_elev, k=3)
    y_smooth = spline(dist_smooth)
    
    plt.plot(dist_smooth, y_smooth, label=f'Сплайн ({k} вузлів)')

plt.title("Профіль висоти: Залежність від кумулятивної відстані")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Графік 2: Похибки
plt.subplot(2, 1, 2)

for k in knot_counts:
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    k_dist = distances[idx]
    k_elev = elevations[idx]
    
    spline = make_interp_spline(k_dist, k_elev, k=3)
    y_pred = spline(distances)
    error = elevations - y_pred
    
    plt.plot(distances, error, marker='o', markersize=4, label=f'Похибка ({k} вузлів)')

plt.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
plt.title("Графік похибок інтерполяції")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Похибка (м)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Збереження графіку
plot_file = "elevation_analysis_plot.png"
plt.savefig(plot_file)
plt.show()