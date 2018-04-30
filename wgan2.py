"""
Implementation of WGAN-GP

https://github.com/OctThe16th/WGAN-GP-with-keras-for-text/blob/master/Exploration/GenerativeAdverserialWGAN-GP.py
"""
from functools import partial

from keras import backend
from keras.layers import (Input, Dense, Dropout,
                          Flatten, Reshape, Subtract)

from keras.layers.merge import _Merge
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

from blocks import (bn_dense,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)
from loss import wasserstein_loss, gradient_penalty_loss


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        real, fake = inputs
        weights = backend.random_uniform((backend.shape(real)[0], 1, 1, 1))
        return (weights * real) + ((1 - weights) * fake)


class WGAN_GP(object):
    def __init__(self, input_dim, img_dim,
                 g_hidden=16,
                 d_hidden=16,
                 gen_opt=Adam,
                 gen_learning_rate=0.0001,
                 critic_opt=Adam,
                 critic_learning_rate=0.0001,
                 train_ratio=5,
                 grad_penalty=10):
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
        self.grad_penalty = grad_penalty

        # Component networks.
        self.generator = self._construct_generator()
        self.critic = self._construct_critic()
        self.gan = self._construct_gan()
        self.training_critic = self._construct_training_critic()

    def _construct_generator(self):
        z = Input(shape=(self.in_dim,))
        z1 = bn_dense(z, 512)
        z_reshp = Reshape((1, 1, 512))(z1)
        deconv_block = deconvnet(z_reshp, self.img_dim, self.g_hidden,
                                 activation='selu', batchnorm=True,
                                 bias=False)
        gen_img = bn_deconv_layer(deconv_block, self.img_dim[-1], 4, 2,
                                  activation='tanh', batchnorm=False,
                                  use_bias=False)

        generator = Model(z, gen_img)
        # generator.compile(optimizer=self.critic_opt(lr=self.critic_lr),
        #                   loss='mse')
        return generator

    def _construct_critic(self):
        img = Input(shape=self.img_dim)
        d = convnet(img, self.d_hidden, batchnorm=False,
                    activation='selu', bias=False)
        d_flat = Flatten()(d)
        d_fc = bn_dense(d_flat, 1024, batchnorm=False,
                        activation='selu', use_bias=False)
        disc_out = Dense(1, use_bias=False)(d_fc)

        critic = Model(img, disc_out)
        # critic.compile(optimizer=self.critic_opt(lr=self.critic_lr),
        #                loss=wasserstein_loss)
        return critic

    def _construct_training_critic(self):
        for l in self.generator.layers:
            l.trainable = False
        for l in self.critic.layers:
            l.trainable = True
        self.generator.trainable = False
        self.critic.trainable = True

        real_img = Input(shape=self.img_dim)
        disc_real_out = self.critic(real_img)

        fake_input = Input(shape=(self.in_dim,))
        fake_img = self.generator(fake_input)
        disc_fake_out = self.critic(fake_img)

        averaged_samples = RandomWeightedAverage()([real_img, fake_img])
        averaged_samples_out = self.critic(averaged_samples)

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  penalty_weight=self.grad_penalty)
        partial_gp_loss.__name__ = 'gradient_penalty'

        discriminator = Model(inputs=[real_img, fake_input],
                              outputs=[disc_real_out,
                                       disc_fake_out,
                                       averaged_samples_out])
        discriminator.compile(optimizer=self.critic_opt(lr=self.critic_lr,
                                                        beta_1=0.5,
                                                        beta_2=0.9),
                              loss=[wasserstein_loss,
                                    wasserstein_loss,
                                    partial_gp_loss])
        return discriminator

    def _construct_gan(self):
        for l in self.generator.layers:
            l.trainable = True
        for l in self.critic.layers:
            l.trainable = False
        self.generator.trainable = True
        self.critic.trainable = False

        z = Input(shape=(self.in_dim,))
        gan = Model(z, self.critic(self.generator(z)))
        gan.compile(optimizer=self.gen_opt(lr=self.gen_lr,
                                           beta_1=0.5, beta_2=0.9),
                    loss=wasserstein_loss)
        return gan

    def _prep_fake(self, batch_size):
        z_fake = np.random.normal(
            0., 1.,[batch_size, self.in_dim]).astype(np.float32)
        return z_fake

    def train_on_batch(self, x_train):
        batch_size = x_train.shape[0]

        # 1. Train Critic (discriminator).
        minibatch_size = batch_size // self.train_ratio
        critic_loss = []

        pos_y = np.ones((minibatch_size, 1), dtype=np.float32)
        neg_y = -pos_y
        dummy_y = np.zeros((minibatch_size, 1), dtype=np.float32)

        for i in xrange(self.train_ratio):
            x_minibatch = x_train[i * minibatch_size:
                                  (i + 1) * minibatch_size,
                                  :, :, :]
            z_fake = self._prep_fake(minibatch_size)
            critic_loss.append(
                self.training_critic.train_on_batch([x_minibatch, z_fake],
                                                    [pos_y, neg_y, dummy_y]))
        critic_loss = np.mean(critic_loss, axis=0)

        # 2. Train Generator.
        z_fake = self._prep_fake(minibatch_size)
        gen_loss = self.gan.train_on_batch(z_fake, pos_y)

        sum_loss = critic_loss[0] + gen_loss
        return (sum_loss, critic_loss, gen_loss)
