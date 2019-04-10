import numpy as np

def vectorize_sequences(sequences, dimention=10000):
    results = np.zeros(len(sequences), dimention)

    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = model.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape(10000,)))
model.add(layers.Dense(16, activation='rele'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
