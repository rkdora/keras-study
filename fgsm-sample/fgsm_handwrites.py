import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from common.functions import softmax

network = DeepConvNet()

# パラメータの保存
network.load_params("deep_convnet_params.pkl")
print("loaded Network Parameters!")

def generate_adv2(x, label, network, eps=0.01):
    d, g = network.gradient_for_fgsm(x, np.array([label]))
#     plt.imshow(d.reshape(28, 28), 'gray')
#     plt.show()
    p = eps * np.sign(d)
    adv = (x + p).clip(min=0, max=1)
#     plt.imshow(adv.reshape(28, 28), 'gray')
#     plt.show()
    return adv



### 画像データ読み込み、加工

# 画像の入っているフォルダを指定し、中身のファイル名を取得
# filenames = sorted(os.listdir('handwrite_numbers'))
filenames = sorted(os.listdir('handwrite3_numbers'))

# フォルダ内の全画像をデータ化
img_test = np.empty((0, 784))
for filename in filenames:
    # 画像ファイルを取得、グレースケール（モノクロ）にしてサイズ変更
    img = Image.open('handwrite_numbers/' + filename).convert('L')
    # 画像の表示
    # img.show()
    resize_img = img.resize((784, 784))

    img_data256 = np.array([])
    # 64画素の1画素ずつ明るさをプロット
    for y in range(28):
        for x in range(28):
            # 1画素に縮小される範囲の明るさの二乗平均をとり、白黒反転
            # crop()は、画像の一部の領域を切り抜くメソッド。切り出す領域を引数(left, upper, right, lower)（要は左上と右下の座標）で指定する。
            crop = np.asarray(resize_img.crop((x * 28, y * 28, x * 28 + 28, y * 28 + 28)))
            bright = 255 - crop.mean()**2 / 255
            img_data256 = np.append(img_data256, bright)

    img_data = img_data256 / 255

    # 加工した画像データをmnist_edited_imagesに出力する
    # cmap='gray'でグレースケールで表示
    # plt.imshow(img_data.reshape(28, 28), cmap='gray')
    # plt.savefig("/Users/ryuto/works/judge-num/mnist_edited_numbers/edited_" + filename.replace(".png", "") + ".png")

    img_test = np.r_[img_test, img_data.reshape(1, -1)]

# 画像データの正解を配列にしておく
img_ans = []
for filename in filenames:
    img_ans += [int(filename[:1])]

img_ans = np.array(img_ans)

img_test = img_test.reshape(-1, 1, 28, 28)

pred = network.predict(img_test)

pred_label = np.argmax(pred, axis=1)

pred_score = list(map(lambda x:round(x, 2), np.max(softmax(pred), axis=1)))

# 結果の出力
print("判定結果")
print("観測：", img_ans)
print("予測：", pred_label)
# print("信頼度：", pred_score)
print("正答率：", np.sum(pred_label == img_ans)/(img_test.shape[0]))

img_advs = []

for i, x in enumerate(img_test):
    x = x.reshape(1, 1, 28, 28)
    adv = generate_adv2(x, pred_label[i], network, 0.3)
    img_advs.append(adv.reshape(1, 28, 28))

img_advs = np.array(img_advs)

pred_advs = network.predict(img_advs)
pred_advs_label = np.argmax(pred_advs, axis=1)

print("攻撃：", pred_advs_label)
print("正答率：", np.sum(pred_advs_label == img_ans)/(img_test.shape[0]))
