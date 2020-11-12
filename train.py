import os
import time
from dataset import CIFAR
from acgan import ACGAN
from acgangp import ACGANgp
import utils
import argparse
import tensorflow as tf
import datetime
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model", default="acgangp")
parser.add_argument("-z", "--latent_dim", dest="latent_dim", default=128, type=int)
parser.add_argument("-l", "--label_dim", dest="label_dim", default=10, type=int)
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("-e", "--epoch", dest="epoch", default=201, type=int)
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true", default=False)
parser.add_argument("--lambda", dest="lambda_", default=10, type=float)
parser.add_argument("--d_loop", dest="d_loop", default=1, type=int)
parser.add_argument("-lr", "--learning_rate", dest="lr", default=0.0002, type=float)
parser.add_argument("-b1", "--beta1", dest="beta1", default=0., type=float)
parser.add_argument("-b2", "--beta2", dest="beta2", default=0.9, type=float)
parser.add_argument("--net", dest="net", default="resnet", type=str, help="dcnet or resnet")
parser.add_argument("--output_dir", dest="output_dir", type=str, default="./visual")
parser.add_argument("--date_dir", dest="data_dir", type=str, default="./")

args = parser.parse_args(
    # ["--model", "acgangp",
    # "--latent_dim", "256",
    # "--label_dim", "10",
    # "--batch_size", "64",
    # "--epoch", "201",
    # "--soft_gpu",
    # "--lambda", "10",
    # "--d_loop", "1",
    # "-lr", "0.0002",
    # "--beta1", "0",
    # "--beta2", "0.9",
    # "--net", "resnet",
    # # "--output_dir", "./visual",
    # # "--data_dir", "./"]
)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(gan, ds):
    _dir = "{}/{}/{}/model".format(args.output_dir, model_name, date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epoch):
        for t, (real_img, real_img_label) in enumerate(ds):
            d_loss, g_loss = gan.step(real_img, real_img_label)

        if ep % 5 == 0:
            utils.save_gan(gan, "%s/ep-%03d" % (date_str, ep), args.output_dir)
        t1 = time.time()
        logger.info("ep={:03d} | time={:05.1f} | d_loss={:.2f} | g_loss={:.2f}".format(
            ep, t1 - t0, d_loss.numpy(), g_loss.numpy()))
        t0 = t1
        if (ep + 1) % 5 == 0:
            gan.save_weights(checkpoint_path.format(epoch=ep))
    gan.save_weights(checkpoint_path.format(epoch=args.epoch))


def traingp(gan, x, y):
    steps = len(x) // args.batch_size
    _dir = "{}/{}/{}/model".format(args.output_dir, model_name, date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epoch):
        for t in range(steps):
            g_loss = gan.train_g(args.batch_size)
            for _ in range(args.d_loop):
                idx = np.random.randint(0, len(x), args.batch_size)
                img = tf.gather(x, idx)
                label = tf.gather(y, idx)
                w_loss, gp_loss, class_loss = gan.train_d(img, label)

        if ep % 5 == 0:
            utils.save_gan(gan, "%s/ep-%03d" % (date_str, ep), args.output_dir)
        t1 = time.time()
        logger.info("ep={:03d} | time={:05.1f} | w_loss={:.2f} | gp={:.2f} | class={:.2f} | g_loss={:.2f}".format(
            ep, t1-t0, w_loss.numpy(), gp_loss.numpy(), class_loss.numpy(), g_loss.numpy()))
        t0 = t1
        if (ep+1) % 5 == 0:
            gan.save_weights(checkpoint_path.format(epoch=ep))
    gan.save_weights(checkpoint_path.format(epoch=args.epoch))


def init_logger(model_name, date_str, m):
    logger = utils.get_logger(model_name, date_str)
    logger.info(str(args))
    logger.info("x_shape: {} | x_type: {} |  y_shape: {} |  y_type: {}".format(
        x_train.shape, x_train.dtype, y_train.shape, y_train.dtype))
    logger.info("model parameters: g={}, d={}".format(m.g.count_params(), m.d.count_params()))

    try:
        tf.keras.utils.plot_model(m.g, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="{}/{}/{}/net_g.png".format(args.output_dir, model_name, date_str))
        tf.keras.utils.plot_model(m.d, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="{}/{}/{}/net_d.png".format(args.output_dir, model_name, date_str))
    except Exception as e:
        print(e)
    return logger


if __name__ == "__main__":
    utils.set_soft_gpu(args.soft_gpu)
    cifar = CIFAR(n_class=args.label_dim)
    (x_train, y_train), (x_test, y_test) = cifar.load()
    print("x_shape:", x_train.shape, " x_type:", x_train.dtype,
          " y_shape:", y_train.shape, " y_type:", y_train.dtype)
    model_name = args.model
    summary_writer = tf.summary.create_file_writer('{}/{}/{}'.format(args.output_dir, model_name, date_str))
    if model_name == "acgan":
        d = utils.get_ds(args.batch_size // 2, x_train, y_train)
        m = ACGAN(args.latent_dim, args.label_dim, x_train.shape[1:], a=-1, b=1, c=1,
                  summary_writer=summary_writer, lr=args.lr, beta1=args.beta1, beta2=args.beta2, net=args.net)
        logger = init_logger(model_name, date_str, m)
        train(m, d)
    elif model_name == "acgangp":
        x_train, y_train = utils.convert_to_tensor(x_train, y_train)
        m = ACGANgp(args.latent_dim, args.label_dim, x_train.shape[1:], args.lambda_,
                    summary_writer=summary_writer, lr=args.lr, beta1=args.beta1, beta2=args.beta2, net=args.net)
        logger = init_logger(model_name, date_str, m)
        traingp(m, x_train, y_train)
    else:
        raise ValueError("model name error")
