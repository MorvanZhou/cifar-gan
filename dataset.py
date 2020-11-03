import pickle
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import os
import shutil
import requests
import re

DEFAULT_DATA_DIR = "./tmp_data"
# CIFAR10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_URL = "http://42.194.138.114/resources/data/image/cifar-10-python.tar.gz"
# CIFAR100_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_URL = "http://42.194.138.114/resources/data/image/cifar-100-python.tar.gz"


class CIFAR:
    def __init__(self, n_class):
        assert n_class in [10, 100], ValueError
        self.n_class = n_class
        self.gz = "cifar-{}-python.tar.gz".format(n_class)

    @property
    def classes_zh(self):
        if self.n_class == 10:
            return [
                "飞机", "小车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"
            ]
        else:
            return [
                "苹果",
                "水族馆鱼",
                "宝宝",
                "熊",
                '海狸',
                '床',
                '蜜蜂',
                '甲虫',
                '自行车',
                '瓶子',
                '碗',
                '男孩',
                '桥',
                '巴士',
                '蝴蝶',
                '骆驼',
                '罐子',
                '城堡',
                '毛虫',
                '牛',
                '椅子',
                '黑猩猩',
                '时钟',
                '云',
                '蟑螂',
                '长椅',
                '螃蟹',
                '鳄鱼',
                '杯子',
                '恐龙',
                '海豚',
                '大象',
                '比目鱼',
                '森林',
                '狐狸',
                '女孩',
                '仓鼠',
                '房子',
                '袋鼠',
                '键盘',
                '灯',
                '割草机',
                '豹子',
                '狮子',
                '蜥蜴',
                '龙虾',
                '男人',
                '枫树',
                '摩托车',
                '山',
                '老鼠',
                '蘑菇',
                '橡树',
                '橙子',
                '兰花',
                '獭',
                '榈树',
                '梨',
                '皮卡车',
                '松树',
                '野',
                '盘子',
                '罂粟',
                '豪猪',
                '负鼠',
                '兔子',
                '狸',
                '射线',
                '路',
                '火箭',
                '玫瑰',
                '海',
                '密封',
                '鲨鱼',
                '母老虎',
                '臭鼬',
                '摩天大厦',
                '蜗牛',
                '蛇',
                '蜘蛛',
                '松鼠',
                '电车',
                '向日葵',
                '甜椒',
                '桌子',
                '坦克',
                '电话',
                '电视',
                '老虎',
                '拖拉机',
                '火车',
                '鳟鱼',
                '郁金香',
                '乌龟',
                '衣柜',
                '鲸鱼',
                '柳树',
                '狼',
                '女人',
                '虫',
            ]

    @property
    def classes(self):
        if self.n_class == 10:
            return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        else:
            return [
                'apple', # id 0
                'aquarium_fish',
                'baby',
                'bear',
                'beaver',
                'bed',
                'bee',
                'beetle',
                'bicycle',
                'bottle',
                'bowl',
                'boy',
                'bridge',
                'bus',
                'butterfly',
                'camel',
                'can',
                'castle',
                'caterpillar',
                'cattle',
                'chair',
                'chimpanzee',
                'clock',
                'cloud',
                'cockroach',
                'couch',
                'crab',
                'crocodile',
                'cup',
                'dinosaur',
                'dolphin',
                'elephant',
                'flatfish',
                'forest',
                'fox',
                'girl',
                'hamster',
                'house',
                'kangaroo',
                'computer_keyboard',
                'lamp',
                'lawn_mower',
                'leopard',
                'lion',
                'lizard',
                'lobster',
                'man',
                'maple_tree',
                'motorcycle',
                'mountain',
                'mouse',
                'mushroom',
                'oak_tree',
                'orange',
                'orchid',
                'otter',
                'palm_tree',
                'pear',
                'pickup_truck',
                'pine_tree',
                'plain',
                'plate',
                'poppy',
                'porcupine',
                'possum',
                'rabbit',
                'raccoon',
                'ray',
                'road',
                'rocket',
                'rose',
                'sea',
                'seal',
                'shark',
                'shrew',
                'skunk',
                'skyscraper',
                'snail',
                'snake',
                'spider',
                'squirrel',
                'streetcar',
                'sunflower',
                'sweet_pepper',
                'table',
                'tank',
                'telephone',
                'television',
                'tiger',
                'tractor',
                'train',
                'trout',
                'tulip',
                'turtle',
                'wardrobe',
                'whale',
                'willow_tree',
                'wolf',
                'woman',
                'worm',
            ]

    def download(self):
        url = CIFAR10_URL if self.n_class == 10 else CIFAR100_URL
        r = requests.head(url)
        if "content-disposition" not in r.headers:
            fname = os.path.basename(url)
        else:
            d = r.headers['content-disposition']
            fname = re.findall("filename=(.+)", d)[0]
        self.gz = fname
        if not os.path.exists(self.gz):
            os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
            r = requests.get(url, stream=True)  # stream loading
            with open(self.gz, 'wb') as f:
                print("downloading...")
                for chunk in r.iter_content(chunk_size=512):
                    f.write(chunk)

    def _load_batch(self, path):
        with open(path, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        if self.n_class == 10:
            labels = d[b"labels"]
        else:
            labels = d[b"fine_labels"]
        return d[b"data"].reshape((-1, 3, 32, 32)), labels

    def load(self, data_format="channels_last", keep_tmp=True):
        if not os.path.exists(self.gz):
            self.download()
            self.load()
        tar = tarfile.open(self.gz, "r:gz" if self.gz.endswith(".tar.gz") else "r:")
        tar.extractall(DEFAULT_DATA_DIR)
        tar.close()

        num_train_samples = 50000
        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        if self.n_class == 10:
            for i in range(1, 6):
                fpath = os.path.join(DEFAULT_DATA_DIR, "cifar-10-batches-py", 'data_batch_' + str(i))
                (x_train[(i - 1) * 10000:i * 10000, :, :, :],
                 y_train[(i - 1) * 10000:i * 10000]) = self._load_batch(fpath)
            fpath = os.path.join(DEFAULT_DATA_DIR, "cifar-10-batches-py", 'test_batch')
        else:
            x_train[:], y_train[:] = self._load_batch(os.path.join(DEFAULT_DATA_DIR, "cifar-100-python", 'train'))
            fpath = os.path.join(DEFAULT_DATA_DIR, "cifar-100-python", 'test')

        x_test, y_test = self._load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if data_format == 'channels_last':
            x_train = np.transpose(x_train, (0, 2, 3, 1))
            x_test = np.transpose(x_test, (0, 2, 3, 1))

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        if not keep_tmp:
            shutil.rmtree(DEFAULT_DATA_DIR)
        return (x_train, y_train), (x_test, y_test)


def show_cifar(classes=10):
    _cifar = CIFAR(classes)
    _, test_data = _cifar.load("channels_last")
    rand_n = np.random.randint(0, len(test_data[0]), size=(25,))
    data, labels = test_data[0][rand_n], test_data[1][rand_n]
    plt.figure(1, (10, 10))
    for i in range(5):
        for j in range(5):
            n = i*5+j
            plt.subplot(5, 5, n+1)
            plt.imshow(data[n])
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(str(labels[n]) + " " + _cifar.classes[int(labels[n])])
    plt.show()


cifar10 = CIFAR(n_class=10)
cifar100 = CIFAR(n_class=100)

if __name__ == "__main__":
    show_cifar(classes=100)
