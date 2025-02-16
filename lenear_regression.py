import numpy as np
import matplotlib.pyplot as plt

X = np.array([50, 70, 90, 110, 130]) #площадь
Y = np.array([120, 150, 180, 210, 240]) #стоимость

w = 0
b = 0
learning_rate = 0.0001
epochs = 1000 #количество итераций

for epoch in range(epochs):
    Y_pred = w + X + b
    
    error = Y_pred - Y
    
    dw = (2 / len(X)) * np.sum(error * X)
    db = (2 / len(X)) * np.sum(error)
    
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if epoch % 100 == 0:
        loss = np.mean(error ** 2)
        print(f"Эпоха {epoch}, Loss: {loss:.2f}, w: {w:.2f}, b: {b:.2f}")
print(f"\nИтоговые параметры: w: {w:.2f}, b: {b:.2f}")



# График данных
plt.scatter(X, Y, color='blue', label='Данные')

# График модели
X_line = np.linspace(50, 130, 100)
Y_line = w * X_line + b
plt.plot(X_line, Y_line, color='red', label='Модель')

plt.xlabel('Площадь дома')
plt.ylabel('Стоимость дома')
plt.legend()
plt.show()