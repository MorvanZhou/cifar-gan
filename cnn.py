from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf


W_INIT = keras.initializers.RandomNormal(0, 0.02)


def dc_d(input_shape=(32, 32, 3), use_bn=True):
    model = keras.Sequential()
    # [n, 32, 32, 3]
    model.add(Conv2D(32, 5, strides=2, padding='same', input_shape=input_shape, kernel_initializer=W_INIT))
    if use_bn:
        model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # 16
    model.add(Conv2D(64, 5, strides=2, padding='same', kernel_initializer=W_INIT))
    if use_bn:
        model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # 8
    model.add(Conv2D(128, 5, strides=2, padding='same', kernel_initializer=W_INIT))
    if use_bn:
        model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # 4
    model.add(Flatten())
    model.add(Dense(64, kernel_initializer=W_INIT))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    return model


def dc_g(input_shape):
    return keras.Sequential([
        # [n, latent]
        Dense(4 * 4 * 256, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(),
        ReLU(),
        Reshape((4, 4, 256)),
        # 4
        Conv2DTranspose(128, 2, 2, padding="same", kernel_initializer=W_INIT),
        BatchNormalization(),
        ReLU(),
        # 8
        Conv2DTranspose(64, 5, 2, padding='same', kernel_initializer=W_INIT),
        BatchNormalization(),
        ReLU(),
        # 16
        Conv2DTranspose(32, 5, 2, padding='same', kernel_initializer=W_INIT),
        BatchNormalization(),
        ReLU(),
        # 32
        Conv2D(3, 5, 1, padding="same",  activation=keras.activations.tanh, kernel_initializer=W_INIT)
    ])


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, bottlenecks=2, transpose=False, use_bn=True):
        super().__init__()
        self.bn = keras.Sequential(
            [ResBottleneck(filters, kernel_size, strides, transpose, use_bn)]
        )
        if bottlenecks > 1:
            for _ in range(1, bottlenecks):
                self.bn.add(ResBottleneck(filters, kernel_size, 1, transpose, use_bn))

    def call(self, x, training=None):
        o = self.bn(x, training=training)
        return o


class ResBottleneck(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, transpose=False, use_bn=True):
        super().__init__()
        self.transpose = transpose

        # 1
        c = filters // 4
        self.b = keras.Sequential([Conv2D(c, 1, strides=1, padding="same", kernel_initializer=W_INIT)])
        if use_bn:
            self.b.add(BatchNormalization())
        self.b.add(self.get_relu())

        # 2
        self.b.add(Conv2D(c, kernel_size, strides=1, padding="same", kernel_initializer=W_INIT))
        if use_bn:
            self.b.add(BatchNormalization())
        self.b.add(self.get_relu())

        # 3
        self.b.add(Conv2D(filters, 1, strides=1, padding="same", kernel_initializer=W_INIT))
        if use_bn:
            self.b.add(BatchNormalization())

        # projection
        self.project = None
        if strides != 1 or strides != (1, 1):
            p = Conv2DTranspose if transpose else Conv2D
            self.project = keras.Sequential([p(filters, 1, strides, padding="same", kernel_initializer=W_INIT)])
            if use_bn:
                self.project.add(BatchNormalization())

    def get_relu(self):
        return ReLU() if self.transpose else LeakyReLU()

    def call(self, x, training=None):
        o = self.b(x, training=training)
        if self.project is not None:
            x = self.project(x, training=training)
        o = self.get_relu()(o + x)
        return o


def resnet_d(input_shape, use_bn=True):
    return keras.Sequential([
        keras.Input(input_shape),   # [32, 32, 3]
        ResBlock(filters=128, strides=2, bottlenecks=1, use_bn=use_bn),  # [16, 16]
        ResBlock(filters=128, strides=2, bottlenecks=1, use_bn=use_bn),  # [8, 8]
        ResBlock(filters=128, strides=1, bottlenecks=1, use_bn=use_bn),  # [8, 8]
        ResBlock(filters=128, strides=1, bottlenecks=1, use_bn=use_bn),  # [8, 8]
        AvgPool2D(8, 8),        # [1, 1]
        Flatten(),
    ], name="resnet")


def resnet_g(input_shape, use_bn=True):
    return keras.Sequential([
        keras.Input(input_shape),   # [10 + 100]
        Dense(4 * 4 * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(),
        Reshape((4, 4, 128)),
        ResBlock(filters=128, bottlenecks=1, transpose=True, use_bn=use_bn),  # [8, 8]
        ResBlock(filters=128, bottlenecks=1, transpose=True, use_bn=use_bn),  # [16, 16]
        ResBlock(filters=128, bottlenecks=1, transpose=True, use_bn=use_bn),  # [32, 32]
        Conv2D(filters=3, kernel_size=5, strides=1, padding="same", activation=keras.activations.tanh),  # [32, 32]
    ], name="resnet")


class ResBlock2(keras.layers.Layer):
    def __init__(self, filters, activation=None, bottlenecks=2, use_bn=True):
        super().__init__()
        self.activation = activation
        self.bn = keras.Sequential(
            [ResBottleneck2(filters, use_bn)]
        )
        if bottlenecks > 1:
            self.bn.add(ReLU())
            for _ in range(1, bottlenecks):
                self.bn.add(ResBottleneck2(filters, use_bn))

    def call(self, x, training=None):
        o = self.bn(x, training=training)
        if self.activation is not None:
            o = self.activation(o)
        return o


class ResBottleneck2(keras.layers.Layer):
    def __init__(self, filters, use_bn=True):
        super().__init__()
        # 1
        c = filters // 4
        self.b = keras.Sequential([Conv2D(c, 1, strides=1, padding="same", kernel_initializer=W_INIT)])
        if use_bn:
            self.b.add(BatchNormalization())
        self.b.add(ReLU())
        self.b.add(Conv2D(filters, 3, strides=1, padding="same", kernel_initializer=W_INIT))
        if use_bn:
            self.b.add(BatchNormalization())

    def call(self, x, training=None):
        o = self.b(x, training=training)
        o = o + x
        return o


def resnet_g2(input_shape, img_shape=(32, 32, 3), use_bn=True):
    h, w = img_shape[0], img_shape[1]
    _h, _w = 4, 4
    m = keras.Sequential([
        keras.Input(input_shape),   # [10 + 100]
        Dense(_h * _w * 128, input_shape=input_shape, kernel_initializer=W_INIT),
        BatchNormalization(),
        Reshape((_h, _w, 128)),
    ], name="resnet")

    while True:
        up_size = [1, 1]
        if _h < h:
            _h *= 2
            up_size[0] = 2
        if _w < w:
            _w *= 2
            up_size[1] = 2
        m.add(UpSampling2D(up_size))
        m.add(ResBlock2(filters=128, bottlenecks=1, use_bn=use_bn))
        m.add(ReLU())
        if _w == w and _h == h:
            break

    m.add(Conv2D(3, 5, 1, "same", activation=keras.activations.tanh))
    return m
