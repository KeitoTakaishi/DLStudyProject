'''
1.訓練データのベクトル化、ラベルデータのベクトル化
2.
'''

import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print('train_data num : ' + str(len(train_data)))
print('test_data num : ' + str(len(test_data)))


#データのベクトル化---------------------------------------
def vectorize_sequences(sequences, dimention=10000):
    results = np.zeros((len(sequences), dimention))
    print(str(sequences.shape))

    #sequenceは1次元list
    for i, sequence in enumerate(sequences):
        #print(sequence)
        #print('----------')
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#ラベルのベクトル化----------------------------------------
def to_one_hot(labels, dimention=46):
    results = np.zeros((len(labels), dimention))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#one_hot_train_labels = to_categorical(train_labels)
#one_hot_test_labels = to_categorical(labels)


#modelの構築
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

x_val = x_train[:1000]
practical_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
practical_y_train = one_hot_train_labels[1000:]

history = model.fit(practical_x_train, practical_y_train, epochs=8, batch_size=512, validation_data=(x_val, y_val))

'''
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''

plt.clf()


acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


result = model.evaluate(x_test, one_hot_test_labels)
print(result)

predictions = model.predict(x_test)
print( np.argmax(predictions[0]) )
