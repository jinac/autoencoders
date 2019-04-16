"""
Implementation of WGAN

https://github.com/tjwei/GANotebooks/blob/master/wgan-keras.ipynb
"""
from keras import backend
from keras.layers import (Input, Activation,
                          Dense, Dropout,
                          Flatten, Reshape)

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)
from loss import wasserstein_loss


class WGAN(object):
    def __init__(self, input_dim, img_dim,
                 g_hidden=32,
                 d_hidden=32,
                 gen_opt=RMSprop,
                 gen_learning_rate=0.00005,
                 critic_opt=RMSprop,
                 critic_learning_rate=0.00005,
                 train_ratio=5,
                 clamp=0.01):
        # Define IO dimensions for models.
        self.in_dim = input_dim
        self.img_dim = img_dim

        # Hidden nodes.
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden

        # Optimizer settings.
        self.gen_lr = gen_learning_rate
        self.gen_opt = gen_opt
        self.critic_lr = critic_learning_rate
        self.critic_opt = critic_opt

        self.train_ratio = train_ratio
        self.clamp_low = -np.abs(clamp)
        self.clamp_high = np.abs(clamp)

        # Component networks.
        self.generator = self._construct_generator()
        self.critic = self._construct_critic()
        self.gan = self._construct_gan()

    def _construct_generator(self):
        z = Input(shape=(self.in_dim,))
        z0 = Reshape((1, 1, self.in_dim))(z)
        deconv_block = deconvnet(z0, self.img_dim,
                                 self.g_hidden, bias=False)
        gen_img = bn_deconv_layer(deconv_block, self.img_dim[-1], 4, 2,
                                  activation='sigmoid', batchnorm=False,
                                  use_bias=False)

        generator = Model(z, gen_img)
        generator.compile(optimizer=self.critic_opt(lr=self.critic_lr),
                          # loss='binary_crossentropy')
                          loss='mse')
        return generator

    def _construct_critic(self):
        img = Input(shape=self.img_dim)
        conv_block = convnet(img, self.d_hidden, bias=False)
        d_flat = Flatten()(conv_block)
        d_dense = bn_dense(d_flat, 1024)
        disc_out = bn_dense(d_dense, 1, activation='linear', use_bias=False)

        critic = Model(img, disc_out)
        critic.compile(optimizer=self.critic_opt(lr=self.critic_lr),
                       loss=wasserstein_loss)
        return critic

    def _construct_gan(self):
        self.generator.trainable = True
        self.critic.trainable = False
        gan = Model(self.generator.inputs[0],
                    self.critic(self.generator.output))
        gan.compile(optimizer=self.gen_opt(lr=self.gen_lr),
                    loss=wasserstein_loss)
        return gan

    def _prep_fake(self, batch_size):
        z_fake = np.random.normal(0., 1., [batch_size, self.in_dim])
        x_fake = self.generator.predict_on_batch(z_fake)
        return z_fake, x_fake

    def _clip_weights(self):
        weights = [np.clip(w, self.clamp_low, self.clamp_high)
                   for w in self.critic.get_weights()]
        self.critic.set_weights(weights)
        # for l in self.critic.trainable_weights:
        #     backend.update(l, backend.clip(l, self.clamp_low, self.clamp_high))

    def train_on_batch(self, x_train):
        batch_size = x_train.shape[0]
        ones = np.ones(batch_size)

        # 1. Train Critic (discriminator).
        for _ in xrange(self.train_ratio):
            # Clip weights trick.
            self._clip_weights()

            # Train on real and fakes.
            z_fake, x_fake = self._prep_fake(batch_size)
            x = np.concatenate([x_train, x_fake])
            y = np.concatenate([-ones, ones])
            critic_loss = self.critic.train_on_batch(x, y)
            # critic_loss = self.critic.train_on_batch(x_train, ones)

            # Train on fakes.
            # critic_loss -= self.critic.train_on_batch(x_fake, -ones)


        # 2. Train Generator.
        z_fake = np.random.normal(0., 1., [batch_size, self.in_dim])
        gen_loss = self.gan.train_on_batch(z_fake, ones)

        # critic_loss = 0.5 * critic_loss
        sum_loss = critic_loss + gen_loss
        return (sum_loss, critic_loss, gen_loss)
