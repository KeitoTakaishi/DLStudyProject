# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist


print(type(load_mnist(flatten=True, normalize=False)))
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


print(x_train.shape)#(60000, 768)
print(t_train.shape)
print(x_test.shape)#(10000, 768)
print(t_test.shape)
