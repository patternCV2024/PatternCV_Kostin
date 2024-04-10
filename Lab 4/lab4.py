#1

import numpy as np

x_train_9 = np.array([[29, 28], [48,  9], [8, 43], [14, 13], [12, 18], [16, 47], [21, 36], [11, 45], [34, 22], [42, 17]])
y_train_9 = np.array([-1, 1, -1, -1, 1, -1, 1, -1, 1, 1])

mw1_9, ml1_9 = np.mean(x_train_9[y_train_9 == 1], axis=0)
mw_1_9, ml_1_9 = np.mean(x_train_9[y_train_9 == -1], axis=0)

sw1_9, sl1_9 = np.var(x_train_9[y_train_9 == 1], axis=0)
sw_1_9, sl_1_9 = np.var(x_train_9[y_train_9 == -1], axis=0)

print('Середнє: ', mw1_9, ml1_9, mw_1_9, ml_1_9)
print('Дисперсії:', sw1_9, sl1_9, sw_1_9, sl_1_9)

x_9 = [40, 10]  # довжина, ширина жука

a_1_9 = lambda x: -(x[0] - ml_1_9) ** 2 / (2 * sl_1_9) - (x[1] - mw_1_9) ** 2 / (2 * sw_1_9)  # Перший класифікатор
a1_9 = lambda x: -(x[0] - ml1_9) ** 2 / (2 * sl1_9) - (x[1] - mw1_9) ** 2 / (2 * sw1_9)  # Другий класифікатор
y_9 = np.argmax([a_1_9(x_9), a1_9(x_9)])  # Обираємо максимум

print('Номер класу (1 - гусениця, -1 - божа корівка): ', y_9)

#2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.7
sigma_x1_squared = 1.9
mu_x1 = [2, 4]
sigma_y1_squared = 0.8
mu_y1 = [2, 4]

# Вхідні параметри для другого кластеру (за замовчуванням)
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([-2, -2])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()

# 2.1

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = -0.7  # Змінений знак коефіцієнта кореляції
sigma_x1_squared = 1.9
mu_x1 = [2, 4]
sigma_y1_squared = 0.8
mu_y1 = [2, 4]

# Вхідні параметри для другого кластеру (за замовчуванням)
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([0, -4])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()

# 2.2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.7
sigma_x1_squared = 1.9
mu_x1 = [2, 4]
sigma_y1_squared = 0.8
mu_y1 = [2, 4]

# Вхідні параметри для другого кластеру
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# Вхідні параметри для третього кластеру
rho3 = -0.5
sigma_x3_squared = 1.5
mu_x3 = [-4, 0]
sigma_y3_squared = 1.5
mu_y3 = [-4, 0]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T
x3 = np.random.multivariate_normal(mu_x3, [[sigma_x3_squared, rho3], [rho3, sigma_y3_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 1/3, 1  # ймовірності появи класів
Py2, L2 = 1/3, 1  
Py3, L3 = 1/3, 1  

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([0, -4])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2), b(x, VV3, mm3, L3, Py3)])  # класифікатор
print("Cluster:", a)

# виведення графіків
plt.figure(figsize=(6, 6))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}, rho3 = {rho3}")
plt.scatter(x1[0], x1[1], s=10, label='Cluster 1')
plt.scatter(x2[0], x2[1], s=10, label='Cluster 2')
plt.scatter(x3[0], x3[1], s=10, label='Cluster 3')
plt.legend()
plt.show()