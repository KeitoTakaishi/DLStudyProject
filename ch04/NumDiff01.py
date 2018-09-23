import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f ,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_diff_dummy(f, x):
    h = 1e-4
    return ((f(x+h) - f(x)) / h)

def func(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = func(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()

grad = numerical_diff(func, 5)
print(grad)
grad = numerical_diff(func, 10)
print(grad)

#------------------------------
grad = numerical_diff_dummy(func, 5)
print(grad)
grad = numerical_diff_dummy(func, 10)
print(grad)
