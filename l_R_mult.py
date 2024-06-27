import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def cost(x, y, w, b):
    cost = 0
    m = x.shape[0]
    for i in range(m):
        tmp_cost = (np.dot(x[i], w) + b) - y[i]
        cost += tmp_cost ** 2
    cost = (1 / (2 * m)) * cost
    return cost

def derivatives(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        error = (np.dot(x[i], w) + b) - y[i] # get error for ith x
        for j in range(n):
            dj_dw[j] += error * x[i, j] # find jth w 
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw

def gradient(x, y, alpha, iterations):
    m, n = x.shape
    w = np.zeros(n)
    b = 0
    for _ in range(iterations):
        dj_db, dj_dw = derivatives(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

def predict(x, w, b):
    return np.dot(x, w) + b

def main():
    iterations = 1000
    alpha = 5.0e-7
    w, b = gradient(X_train, y_train, alpha, iterations)
    f_wb = predict(X_train[0], w, b)
    print(f"f_wb = {f_wb}")

main()
