import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from common.trainer import Trainer

network = DeepConvNet()

# パラメータの保存
network.load_params("deep_convnet_params.pkl")
print("loaded Network Parameters!")


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

pred = np.argmax(pred, axis=1)

# 結果の出力
print("判定結果")
print("観測：", img_ans)
print("予測：", pred)
print("正答率：", np.sum(pred == img_ans)/(img_test.shape[0]))
