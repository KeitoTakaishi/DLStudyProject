import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# digit = train_images[4]
# plt.imshow(digit, cmap='GnBu')
#plt.show()

# my_slice = train_images[10:100]
# print(str(my_slice.shape))


def naive_relu(x):
    assert len(x.shape == 2)
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)

x = np.array([[1,1,1], [1, 1, 1]])
y = np.array([[1,1], [1, 1], [1, 1]])
dot=  np.dot(x, y)
print(str(x.shape))
print(str(y.shape))
print(dot)
