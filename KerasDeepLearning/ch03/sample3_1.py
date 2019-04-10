'''
layer num = 1 epochs = 20 : [0.4957116237640381, 0.85488]
layer num = 1 epochs = 4 : [0.3189006894493103, 0.87268]

layer num = 2 epochs = 20 : [0.7439062911558151, 0.84984]
layer num = 2 epochs = 4 : [0.3129247725868225, 0.87392]

layer num = 3 epochs = 20 : [0.8126771096968651, 0.84856]
layer num = 3 epochs = 4 : [0.3284707674694061, 0.86552]
'''
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
'''
word_index.items()は
('lance', 6435)
("pipe's", 88580)
('discretionary', 64179)
('contends', 40829)
('copywrite', 88581)
('geysers', 52003)
('artbox', 88582)
('cronyn', 52004)
('hardboiled', 52005)
("voorhees'", 88583)
みたいな感じでwordとindexが対になっている辞書型変数
'''

#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorize_sequences(sequences, dimention=10000):
    results = np.zeros((len(sequences), dimention))

    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

#訓練データのベクトル化
x_train = vectorize_sequences(train_data)
#テストデータのベクトル化
x_test = vectorize_sequences(test_data)

#ラベルのベクトル化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#訓練データ
x_val = x_train[:10000]
practical_x_train = x_train[:10000]

#ラベルデータ
y_val = y_train[:10000]
practical_y_train = y_train[:10000]

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(practical_x_train, practical_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
result = model.evaluate(x_test, y_test)
print(result)

print('----------------------')

print(model.predict(x_test))



history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

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
