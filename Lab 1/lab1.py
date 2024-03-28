# Варіант 9
import numpy as np
import matplotlib.pyplot as plt

# Навчальні дані
x_train_9 = np.array([[16, 9],
                    [38, 44],
                    [29, 14],
                    [41, 16],
                    [50, 19],
                    [48, 11],
                    [11, 43],
                    [21, 42],
                    [41, 39],
                    [34, 46]])
y_train = np.array([-1, -1, -1, -1, 1, 1, -1, -1, -1, 1])

n_train = len(x_train)                          # Розмір навчальної вибірки
w = [0, -1]                                     # Початкове значення ваги w
a = lambda x: np.sign(x[0]*w[0] + x[1]*w[1])    # Правило класифікації
L = 0.1                                         # Крок зміни ваги
e = 0.1                                         # Невеликий додаток до w0, щоб забезпечити зазор між лінією розділення та областю
count = 0

last_error_index = -1                           # Індекс останньої помилкової спостереження

# Тренування моделі
for _ in range(100):
    c += 1
    for i in range(n_train):                # Ітерація по спостереженням
        if y_train[i]*a(x_train[i]) < 0:    # Якщо помилка класифікації,
            w[0] = w[0] + L * y_train[i]    # То коригування ваги w0
            last_error_index = i

    Q = sum([1 for i in range(n_train) if y_train[i]*a(x_train[i]) < 0])
    if Q == 0:      # Показник якості класифікації (кількість помилок)
        break       # Зупинка, якщо всі класифікуються правильно

if last_error_index > -1:
    w[0] = w[0] + e * y_train[last_error_index]

print(w)

# Визначення координат для лінії розділення
line_x = np.linspace(0, 50, 100)
line_y = -(w[0]*line_x)/w[1]

# Розділення даних по класам
x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

# Візуалізація результатів
plt.scatter(x_0[:, 0], x_0[:, 1], color='red', label='Class 1')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue', label='Class -1')
plt.plot(line_x, line_y, color='green', label='Decision Boundary')

plt.xlim([0, 50])
plt.ylim([0, 50])
plt.ylabel("довжина")
plt.xlabel("ширина")
plt.legend()
plt.grid(True)
plt.show()