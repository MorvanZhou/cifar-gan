import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys


def set_soft_gpu(soft_gpu):
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def save_gan(model, ep, output_dir="./visual", show_label=False):
    model_name = model.__class__.__name__.lower()
    img_label = np.arange(0, 10).astype(np.int32).repeat(10, axis=0)
    imgs = model.call(img_label, training=False)

    imgs = (imgs + 1) / 2
    plt.clf()
    nc, nr = 10, 10
    plt.figure(0, (nc * 2, nr * 2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nr, nc, i + 1)
            plt.imshow(imgs[i])
            plt.axis("off")
            if show_label:
                plt.text(25, 25, int(r), fontsize=23)
    plt.tight_layout()
    path = "{}/{}/{}.png".format(output_dir, model_name, ep)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def convert_to_tensor(x, y):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if isinstance(y, np.ndarray):
        y = tf.convert_to_tensor(y, dtype=tf.int32)
    x = tf.cast(x, tf.float32) / 255. * 2 - 1
    y = tf.squeeze(tf.cast(y, tf.int32), axis=1)
    return x, y


def get_ds(batch_size, x, y):
    x, y = convert_to_tensor(x, y)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).cache().shuffle(1024).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_logger(model_name, date_str):
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = "visual/{}/{}/train.log".format(model_name, date_str)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(log_fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger