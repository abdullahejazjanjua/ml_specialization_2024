import numpy as np
import matplotlib.pyplot as plt

# Corrected training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_model(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

w = 200
b = 100

f_wb = compute_model(x_train, w, b)

plt.plot(x_train, f_wb, c="r")
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.xlabel('Price')
plt.ylabel('Size')
plt.title('Linear Model')
plt.show()

x_i = 1.2
y_predict = w * x_i + b

print(f"Price is: {y_predict}")