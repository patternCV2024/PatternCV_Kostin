#1
import numpy as np

# Функція для передбачення значень поліноміальної моделі
def predict_poly(x, koeff):
    res = 0
    # Обчислення значень для кожного степеня x та відповідного коефіцієнта
    xx = [x ** (len(koeff) - n - 1) for n in range(len(koeff))]

    # Обчислення значення полінома за формулою
    for i, k in enumerate(koeff):
        res += k * xx[i]

    return res

# Створення даних для x та y (змінена функція y)
x = np.arange(0, 10.1, 0.1)
y = np.exp(-x) * np.cos(x) # змінена функція

# Обрання кожної другої точки для тренувальних даних
x_train, y_train = x[::2], y[::2]

N = len(x)

# Підгонка полінома 10-го степеня до тренувальних даних
z_train = np.polyfit(x_train, y_train, 10)
print(z_train)

#2
import numpy as np
import matplotlib.pyplot as plt

# Дані
x_train_9 = np.array([[5, 49], [39, 16], [49, 10], [7, 37], [9, 37], [41, 42], [8, 20], [13, 27], [17, 32], [19, 12]])
y_train_9 = np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1])

# Параметри моделі
N = 13  # розмір простору ознак (степінь полінома N-1)
L = 20  # Параметр регуляризації

# Матриця вхідних векторів
X_train_9 = np.array([[a ** n for n in range(N)] for a in x_train_9[:, 0]])

# Матриця Y для вихідних даних
Y_train_9 = y_train_9

# Матриця lambda*I
IL = np.array([[L if i == j else 0 for j in range(N)] for i in range(N)])
IL[0][0] = 0  # перший коефіцієнт не регуляризується

# Обчислення коефіцієнтів за формулою w = (XT*X + lambda*I)^-1 * XT * Y
A = np.linalg.inv(X_train_9.T @ X_train_9 + IL)
w = Y_train_9 @ X_train_9 @ A
print(w)

# Відображення графіку
x_range = np.linspace(0, 50, 1000)
X_range = np.array([[a ** n for n in range(N)] for a in x_range])

yy = [np.dot(w, x) for x in X_range]
plt.plot(x_range, yy, label='Прогноз моделі')
plt.scatter(x_train_9[:, 0], y_train_9, c='r', label='Точки даних')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Поліноміальна регресія з L2-регуляризатором')
plt.legend()
plt.grid(True)
plt.show()

#3
import numpy as np
import matplotlib.pyplot as plt

# Сигмоїдна функція втрат
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# Похідна від сигмоїдальної функції втрат по вектору w
def df(w, x, y):
    L1 = 1.0  # Коефіцієнт L1-регуляризатора
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + L1 * np.sign(w)

# Навчальна вибірка (з вашого варіанту)
x_train_9 = np.array([[5, 49], [39, 16], [49, 10], [7, 37], [9, 37], [41, 42], [8, 20], [13, 27], [17, 32], [19, 12]])
x_train_9 = np.hstack((x_train_9, np.ones((len(x_train_9), 1))))  # Додаємо стовпець константи
y_train_9 = np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1])

fn = len(x_train_9[0])
n_train = len(x_train_9)  # Розмір навчальної вибірки
w = np.zeros(fn)        # Початкові вагові коефіцієнти
nt = 0.00001            # Крок збіжності SGD
lm = 0.01               # Швидкість "забування" для Q
N = 5000                # Кількість ітерацій SGD

Q = np.mean([loss(w, x, y) for x, y in zip(x_train_9, y_train_9)])  # Показник якості
Q_plot = [Q]

# Стохастичний алгоритм градієнтного спуску
for i in range(N):
    k = np.random.randint(0, n_train - 1)       # Випадковий індекс
    ek = loss(w, x_train_9[k], y_train_9[k])        # Визначення втрат для обраного вектора
    w = w - nt * df(w, x_train_9[k], y_train_9[k])  # Коригування вагів за допомогою SGD
    Q = lm * ek + (1 - lm) * Q                  # Перерахунок показника якості
    Q_plot.append(Q)

Q = np.mean([loss(w, x, y) for x, y in zip(x_train_9, y_train_9)]) # Справжнє значення емпіричного ризику після навчання
print("Вагові коефіцієнти:", w)
print("Показник якості:", Q)

# Відображення графіка показника якості
plt.plot(Q_plot)
plt.grid(True)
plt.xlabel('Ітерації')
plt.ylabel('Показник якості')
plt.title('Динаміка показника якості під час навчання')
plt.show()

#4
import numpy as np
import matplotlib.pyplot as plt

# Сигмоїдна функція втрат
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# Похідна від сигмоїдальної функції втрат по вектору w
def df(w, x, y):
    L2 = 0.01  # Коефіцієнт L2-регуляризатора
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + 2 * L2 * w

# Навчальна вибірка з трьома ознаками (третій - константа +1)
x_train = np.array([[5, 49], [39, 16], [49, 10], [7, 37], [9, 37], [41, 42], [8, 20], [13, 27], [17, 32], [19, 12]])
x_train = np.hstack((x_train, np.ones((len(x_train), 1))))  # Додаємо стовпець константи
y_train = np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1])

fn = len(x_train[0])
n_train = len(x_train)  # Розмір навчальної вибірки
w = np.zeros(fn)        # Початкові вагові коефіцієнти
nt = 0.00001            # Крок збіжності SGD
lm = 0.01               # Швидкість "забування" для Q
N = 5000                # Кількість ітерацій SGD

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])  # Показник якості
Q_plot = [Q]

# Стохастичний алгоритм градієнтного спуску
for i in range(N):
    k = np.random.randint(0, n_train - 1)       # Випадковий індекс
    ek = loss(w, x_train[k], y_train[k])        # Визначення втрат для обраного вектора
    w = w - nt * df(w, x_train[k], y_train[k])  # Коригування вагів за допомогою SGD
    Q = lm * ek + (1 - lm) * Q                  # Перерахунок показника якості
    Q_plot.append(Q)

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)]) # Справжнє значення емпіричного ризику після навчання
print("Вагові коефіцієнти:", w)
print("Показник якості:", Q)

# Відображення графіка показника якості
plt.plot(Q_plot)
plt.grid(True)
plt.xlabel('Ітерації')
plt.ylabel('Показник якості')
plt.title('Динаміка показника якості під час навчання')
plt.show()