import tensorflow as tf
from tensorflow import keras
import numpy as np
from cnn import dc_d as d_net, resnet_g2 as g_net


class ACGANgp(keras.Model):
    def __init__(self, latent_dim, label_dim, img_shape, lambda_=10, k=1, summary_writer=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_shape = img_shape
        self.lambda_ = lambda_
        self.k = k
        self.g = self._get_generator()
        self.d = self._get_discriminator()

        # beta_1 = 0.5 gives bad results
        self.opt = keras.optimizers.Adam(0.0002, beta_1=0, beta_2=0.9)
        self.loss_class = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.summary_writer = summary_writer
        self._train_step = 0

    def call(self, target_labels, training=None, mask=None):
        noise = tf.random.normal((len(target_labels), self.latent_dim))
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        return self.g.call([noise, target_labels], training=training)

    def _get_discriminator(self):
        img = keras.Input(shape=self.img_shape)
        s = keras.Sequential([
            d_net(self.img_shape, use_bn=False),
            keras.layers.Dense(1+self.label_dim),
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
        s = g_net((self.latent_dim+self.label_dim,))
        o = s(model_in)
        model = keras.Model([noise, label], o, name="generator")
        model.summary()
        return model

    # gradient penalty
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d.call(noise_img, training=True)
        g = tape.gradient(o, noise_img)  # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - self.k)
        return tf.reduce_mean(gp)

    def train_d(self, img, img_label):
        g_img = self.call(img_label, training=False)
        gp = self.gp(img, g_img)
        all_img = tf.concat((img, g_img), axis=0)
        all_img_label = tf.concat((img_label, img_label), axis=0)
        with tf.GradientTape() as tape:
            pred, pred_class = self.d.call(all_img, training=True)
            loss_class = self.loss_class(all_img_label, pred_class)
            pred_real, pred_fake = pred[:len(img)], pred[len(img):]
            w_distance = tf.reduce_mean(pred_real) - tf.reduce_mean(pred_fake)  # maximize W distance
            gp_loss = self.lambda_ * gp
            loss = (gp_loss + loss_class - w_distance) / 3
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d_w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("d_gp", gp_loss, step=self._train_step)
                tf.summary.scalar("d_crossentropy", loss_class, step=self._train_step)
                tf.summary.histogram("d_pred_real", pred_real, step=self._train_step)
                tf.summary.histogram("d_pred_fake", pred_fake, step=self._train_step)
        return w_distance, gp_loss, loss_class

    def train_g(self, batch_size):
        random_img_label = tf.convert_to_tensor(
            np.random.randint(0, self.label_dim, batch_size), dtype=tf.int32)
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred_fake, pred_class = self.d.call(g_img, training=False)
            loss_class = self.loss_class(random_img_label, pred_class)
            w_distance = tf.reduce_mean(-pred_fake)  # minimize W distance
            loss = w_distance + loss_class
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g_w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("g_crossentropy", loss_class, step=self._train_step)
                tf.summary.histogram("g_pred_fake", pred_fake, step=self._train_step)
                if self._train_step % 1000 == 0:
                    tf.summary.image("g_img", (g_img + 1) / 2, max_outputs=5, step=self._train_step)
        self._train_step += 1
        return loss
