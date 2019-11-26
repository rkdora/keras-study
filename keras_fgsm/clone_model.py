import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import RMSprop
from fgsm.deep_convnet import DeepConvNet
from common.functions import softmax
from common.trainer import Trainer

num_classes = 10
max_epochs = 5

# データの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()

## 1次元へ整形
# x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)

##  4次元へ整形
x_train_shape = x_train.shape
x_train = x_train.reshape(x_train_shape[0], 1, x_train_shape[1], x_train_shape[2])

x_test_shape = x_test.shape
x_test = x_test.reshape(x_test_shape[0], 1, x_test_shape[1], x_test_shape[2])


# 正規化
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

# keras用に再整形
X_train = x_train.reshape(60000, 784)
X_test  = x_test.reshape(10000, 784)

y_train = keras.utils.to_categorical(t_train, num_classes)
y_test = keras.utils.to_categorical(t_test, num_classes)

# モデルを読み込む
model = model_from_json(open('../keras_sample/mnist_mlp_model.json').read())

# 学習結果を読み込む
model.load_weights('../keras_sample/mnist_mlp_weights.h5')

model.summary();

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss :', score[0])
print('kerasモデルの正答率：', score[1])

# kerasモデルの出力
model_pred = model.predict(X_train)

network = DeepConvNet()
trainer = Trainer(network, x_train, model_pred, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

test_acc = network.accuracy(x_test, t_test)
print("クローンモデルの正答率：", test_acc)

# パラメータの保存
network.save_params("keras_clone_params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
# plt.show()
plt.savefig('keras_clone_graph.png')
