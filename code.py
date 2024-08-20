import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('var5.txt')

data_sorted = np.sort(data)

empirical_survival = np.arange(1, len(data_sorted) + 1) / len(data_sorted) #вычисление эмп.функции выживаемости(относительная частота выживания)

def gompers(t, a, b):
    return np.exp(-a * np.exp(-b * t)) #S(t)= exp(-a * exp(-b * t))

popt, pcov = curve_fit(gompers, data_sorted, empirical_survival, p0=(0.001, 0.001)) #подгонка кривой гомперца к эмпирическим данным методом наименьших квадратов

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(data_sorted, bins=10, density=True, color='skyblue', alpha=0.7)
plt.title('Гистограмма экспериментальных данных')
plt.xlabel('Дни')
plt.ylabel('Вероятность')
plt.grid(True, linestyle='--', alpha=0.5)

# Empirical Survival Function
plt.figure(figsize=(8, 6))
plt.plot(data_sorted, empirical_survival, 'go-', label='Эмпирическая функция дожития')
plt.title('Эмпирическая функция дожития')
plt.xlabel('Дни')
plt.ylabel('Вероятность')
plt.grid(True, linestyle='--', alpha=0.5)

# Fitted Gompertz Curve
T = np.linspace(min(data_sorted), max(data_sorted), 100)
plt.plot(T, gompers(T, *popt), 'b-', label='Кривая Гомперца')

plt.title('Анализ данных с использованием кривой Гомперца')
plt.xlabel('Дни')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

print(f'Параметры кривой Гомперца (a, b): {popt}')
