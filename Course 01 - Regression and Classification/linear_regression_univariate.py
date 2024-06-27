import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_model(x,w,b):

    f_wb = w * x + b
    return f_wb

def compute_gradient(x, y, w, b):
    
    dj_dw = 0
    dj_db = 0
    m = x.shape[0]

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])

    dj_db /= m
    dj_dw /= m

    return dj_dw, dj_db

def gradient_descent(x, y, alpha, iterations):

    w = 0
    b = 0
    
    for i in range(iterations):
        dj_dw , dj_db = compute_gradient(x,y,w,b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b


def main():

    iterations = 10000
    tmp_alpha = 1.0e-2

    w, b = gradient_descent(x_train, y_train,tmp_alpha, iterations)

    print(f"w = {w}")
    print(f"b = {b}")

    f_wb = compute_model(x_train, w, b)
    plt.plot(x_train, f_wb, c="r")
    plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
    plt.xlabel('Price')
    plt.ylabel('Size')
    plt.title('Linear Model')
    plt.show()



main()