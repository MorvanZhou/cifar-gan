import tensorflow as tf
from tensorflow import keras
import numpy as np
from cnn import resnet_g, dc_g, dc_d


class ACGAN(keras.Model):
    """
    discriminator 图片 预测 真假+标签
    generator 标签 生成 图片
    """
    def __init__(self, latent_dim, label_dim, img_shape, a=-1, b=1, c=1, summary_writer=None, lr=0.0002, beta1=0.5, beta2=0.999, net="dcnet"):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_shape = img_shape
        self.a, self.b, self.c = a, b, c
        self.net_name = net

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.loss_mse = keras.losses.MeanSquaredError(reduction="none")
        self.loss_class = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        self._train_step = 0
        self.summary_writer = summary_writer

    def call(self, target_labels, training=None, mask=None):
        noise = tf.random.normal((len(target_labels), self.latent_dim))
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        return self.g.call([noise, target_labels], training=training)

    def _get_discriminator(self):
        net = dc_d(self.img_shape, use_bn=True)
        img = keras.Input(shape=self.img_shape)
        s = keras.Sequential([
            net,
            keras.layers.Dense(1+self.label_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)),
        ], name="s")
        o = s(img)
        o_ls, o_class = o[:, :1], o[:, 1:]
        model = keras.Model(img, [o_ls, o_class], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        noise = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(), dtype=tf.int32)
        label_onehot = tf.one_hot(label, depth=self.label_dim)
        model_in = tf.concat((noise, label_onehot), axis=1)
        net = dc_g if self.net_name == "dcnet" else resnet_g
        s = net((self.latent_dim+self.label_dim,))
        o = s(model_in)
        model = keras.Model([noise, label], o, name="generator")
        model.summary()
        return model

    def train_d(self, img, img_label, label):
        with tf.GradientTape() as tape:
            pred_bool, pred_class = self.d.call(img, training=True)
            loss_mse = self.loss_mse(label, pred_bool)      # ls loss
            loss_class = self.loss_class(img_label, pred_class)
            loss = tf.reduce_mean(loss_mse + loss_class)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d_mse", tf.reduce_mean(loss_mse), step=self._train_step)
                tf.summary.scalar("d_crossentropy", tf.reduce_mean(loss_class), step=self._train_step)
                tf.summary.histogram("g/last_grad", grads[-1], step=self._train_step)
        return loss

    def train_g(self, random_img_label):
        d_label = self.c * tf.ones((len(random_img_label), 1), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred_bool, pred_class = self.d.call(g_img, training=False)
            loss_mse = self.loss_mse(d_label, pred_bool)    # ls loss
            loss_class = self.loss_class(random_img_label, pred_class)
            loss = tf.reduce_mean(loss_mse + loss_class)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g_mse", tf.reduce_mean(loss_mse), step=self._train_step)
                tf.summary.scalar("g_crossentropy", tf.reduce_mean(loss_class), step=self._train_step)
                tf.summary.histogram("g/first_grad", grads[0], step=self._train_step)
                if self._train_step % 1000 == 0:
                    tf.summary.image("g_img", (g_img + 1) / 2, max_outputs=5, step=self._train_step)
        self._train_step += 1
        return loss, g_img

    def step(self, real_img, real_img_label):
        random_img_label = tf.convert_to_tensor(np.random.randint(0, self.label_dim, len(real_img)*2), dtype=tf.int32)
        g_loss, g_img = self.train_g(random_img_label)

        img = tf.concat((real_img, g_img[:len(g_img)//2]), axis=0)
        img_label = tf.concat((real_img_label, random_img_label[:len(g_img) // 2]), axis=0)
        d_label = tf.concat((
            self.b * tf.ones((len(real_img_label), 1), tf.float32),
            self.a * tf.ones((len(g_img)//2, 1), tf.float32)
        ), axis=0)
        d_loss = self.train_d(img, img_label, d_label)
        return d_loss, g_loss

