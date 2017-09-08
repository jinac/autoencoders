"""
Using ideas from https://arxiv.org/pdf/1706.04223.pdf
"""
from keras.layers import (Dense, Dropout, Flatten,
                          Input, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer)


class ARAE(object):
    def __init__(self, input_dim, latent_dim,
                 noise_dim=32,
                 ae_opt=Adam,
                 ae_learning_rate=0.0001,
                 critic_opt=Adam,
                 critic_learning_rate=0.0001,
                 prior_mu=0.,
                 prior_sigma=10.0):
        # Define IO dimensions for models.
        self.in_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

        # Component networks.
        self.encoder = self._construct_encoder(input_dim,
                                               latent_dim)
        self.decoder = self._construct_decoder(latent_dim,
                                               input_dim)
        self.generator = self._construct_generator(noise_dim,
                                                   latent_dim)
        self.critic = self._construct_critic(latent_dim,
                                             critic_opt,
                                             critic_learning_rate)

        self.autoencoder = self._construct_ae(ae_opt, ae_learning_rate)
        self.gan = self._construct_gan(critic_opt, critic_learning_rate)

        # Latent space prior distribution variables.
        self.mu = prior_mu
        self.sigma = prior_sigma

    def _construct_encoder(self, in_dim, out_dim):
        x = Input(shape=in_dim)
        conv_1 = bn_conv_layer(x, 64, (3, 3))
        conv_2 = bn_conv_layer(conv_1, 16, (3, 3))
        conv_3 = bn_conv_layer(conv_2, 8, (3, 3))
        flat_1 = Flatten()(conv_3)
        do = Dropout(0.3)(flat_1)
        fc_1 = bn_dense(do, 128)
        code_real = Dense(out_dim)(fc_1)

        encoder = Model(x, code_real)
        return encoder

    def _construct_decoder(self, in_dim, out_dim):
        code_real = Input(shape=(in_dim,))
        fc_2 = bn_dense(code_real, 128)

        shp = 8 * np.product(out_dim)
        ups_1 = Dense(shp, activation='relu')(fc_2)

        shp = (out_dim[0], out_dim[1], 8)
        reshp_1 = Reshape(shp)(ups_1)
        deconv_1 = bn_deconv_layer(reshp_1, 8, 3)
        deconv_2 = bn_deconv_layer(deconv_1, 16, 3)
        deconv_3 = bn_deconv_layer(deconv_2, 64, 3)
        x_reconst = Conv2DTranspose(out_dim[2], 3,
                                    padding='same',
                                    activation='sigmoid')(deconv_3)

        decoder = Model(code_real, x_reconst)
        return decoder

    def _construct_generator(self, in_dim, out_dim):
        z = Input(shape=(in_dim,))
        fc_3 = bn_dense(z, 64)
        fc_4 = bn_dense(fc_3, 128)
        fc_5 = bn_dense(fc_4, 256)
        code_fake = bn_dense(fc_5, 2)

        generator = Model(z, code_fake)
        return generator

    def _construct_critic(self, in_dim,
                          critic_opt, critic_learning_rate):
        code = Input(shape=(in_dim,))
        fc_6 = bn_dense(code, 64, activation=None)
        lk_act_1 = LeakyReLU(0.2)(fc_6)
        fc_7 = bn_dense(lk_act_1, 32, activation=None)
        lk_act_2 = LeakyReLU(0.2)(fc_7)
        fc_8 = bn_dense(lk_act_2, 32, activation=None)
        lk_act_3 = LeakyReLU(0.2)(fc_8)
        real_prob = bn_dense(lk_act_3, 1, activation='sigmoid')

        critic = Model(code, real_prob)
        critic.compile(optimizer=critic_opt(lr=critic_learning_rate),
                       loss='binary_crossentropy')
        return critic

    def _construct_ae(self, ae_opt, ae_learning_rate):
        """
        Construct AE and compile for training.
        """
        autoencoder = Model(self.encoder.input,
                            self.decoder(self.encoder.output))
        autoencoder.compile(optimizer=ae_opt(lr=ae_learning_rate),
                            loss='binary_crossentropy')
        return autoencoder

    def _construct_gan(self, critic_opt, critic_learning_rate):
        """
        Construct GAN to teach generator to trick critic of latent space.
        """
        self.critic.trainable = False
        gan = Model(self.generator.input, self.critic(self.generator.output))
        gan.compile(optimizer=critic_opt(lr=critic_learning_rate),
                    loss='binary_crossentropy')
        return gan

    def _generate_noise(self, batch_size):
        return self.sigma * np.random.randn(batch_size,
                                            self.noise_dim) + self.mu

    def _prep_data(self, x_train):
        """
        Prep data and create vectors for training.
        """
        batch_size = x_train.shape[0]
        x_pos = self.encoder.predict_on_batch(x_train)
        y_pos = np.ones(batch_size)

        gen_noise = self._generate_noise(batch_size)
        x_neg = self.generator.predict_on_batch(gen_noise)
        y_neg = np.zeros(batch_size)
        y_train = x_train

        return (gen_noise,
                x_neg, y_neg,
                x_pos, y_pos,\
                y_train)

    def train_on_batch(self, x_train):
        # Prep data for batch update.
        (gen_noise,
         x_neg, y_neg,
         x_pos, y_pos,
         y_train) = self._prep_data(x_train)

        # 1. Train Autoencoder.
        ae_loss = self.autoencoder.train_on_batch(x_train, y_train)

        # 2. Train Critic.
        critic_loss = self.critic.train_on_batch(x_neg, y_neg)
        critic_loss += self.critic.train_on_batch(x_pos, y_pos)

        # 3. Train Generator to trick updated critic.
        gen_loss = self.gan.train_on_batch(gen_noise, y_pos)

        sum_loss = ae_loss + critic_loss + gen_loss
        return (sum_loss, ae_loss, critic_loss, gen_loss)
