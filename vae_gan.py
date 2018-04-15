"""
Implementation of VAE-GAN
"""
from keras import backend
from keras.layers import (Input, Activation,
                          Dense, Dropout,
                          Flatten, Lambda,
                          Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.optimizers import Adam, SGD

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

import vae
from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer)


class VAE_GAN(vae.VAE):
    def __init__(self, input_dim, latent_dim,
                 hidden_dim=128,
                 enc_param=64,
                 dec_param=16,
                 ae_opt=Adam,
                 ae_learning_rate=0.0002,
                 mu=0., std_dev=1.,
                 critic_opt=Adam,
                 critic_learning_rate=0.0001):
        # Inherit from base vae.
        super(VAE_GAN, self).__init__(input_dim, latent_dim, hidden_dim,
                                      enc_param, dec_param,
                                      ae_opt, ae_learning_rate,
                                      mu, std_dev)

        # Critic variables.
        self.critic_opt = critic_opt
        self.critic_learning_rate = critic_learning_rate

        # Component networks.
        self.generator = self.decoder
        self.critic = self._construct_critic()
        self.gan = self._construct_gan()


    def _construct_critic(self):
        """
        """
        gen_img = Input(shape=self.img_dim)
        conv_1 = bn_conv_layer(gen_img, 32, 3, activation=None)
        lk_act_1 = LeakyReLU(0.2)(conv_1)
        conv_2 = bn_conv_layer(lk_act_1, 64, 3, activation=None)
        lk_act_2 = LeakyReLU(0.2)(conv_2)
        conv_3 = bn_conv_layer(lk_act_2, 128, 3, activation=None)
        lk_act_3 = LeakyReLU(0.2)(conv_3)
        flat = Flatten()(lk_act_3)
        real_prob = Dense(1, activation='sigmoid')(flat)

        critic = Model(gen_img, real_prob)
        critic.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                       loss='binary_crossentropy')
        return critic

    def _construct_gan(self):
        """
        Construct GAN to teach decoder/generator to trick critic.
        """
        self.critic.trainable = False
        gan = Model(self.generator.inputs[0], self.critic(self.generator.outputs[0]))
        gan.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                    loss='binary_crossentropy')
        return gan

    def _prep_data(self, x_train):
        """
        Prep data and create vectors for training.
        """
        batch_size = x_train.shape[0]
        x_neg = self.autoencoder.predict_on_batch(x_train)
        y_neg = np.zeros(batch_size)
        x_pos = x_train
        y_pos = np.ones(batch_size)
        y_train = x_train

        return (x_neg, y_neg, x_pos, y_pos, y_train)

    def train_on_batch(self, x_train):
        # Prep data for batch update.
        (x_neg, y_neg,
         x_pos, y_pos,
         y_train) = self._prep_data(x_train)

        # 1. Train Autoencoder.
        ae_loss = self.autoencoder.train_on_batch(x_train, y_train)

        # 2. Train Critic (Discriminator).
        critic_loss = self.critic.train_on_batch(x_neg, y_neg)
        critic_loss += self.critic.train_on_batch(x_pos, y_pos)

        # 3. Train Generator.
        gan_loss = self.gan.train_on_batch(
            self.encoder.predict_on_batch(x_train), y_pos)

        sum_loss = ae_loss + critic_loss + gan_loss
        return (sum_loss, ae_loss, critic_loss, gan_loss)
