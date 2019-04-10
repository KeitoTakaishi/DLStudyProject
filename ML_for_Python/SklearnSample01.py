import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()

#targetは正解ラベル
flag_3_8 = (digits.target == 3) + (digits.target == 8)
images = digits[flag_3_8]
labels = digits[flag_3_8]


# for label, img in zip(digits.target[:10], digits.images[:10]):
#     plt.subplot(2, 5, label + 1)
#     plt.axis('off')
#     plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Digit: {0}'.format(label))
#
# plt.show()
