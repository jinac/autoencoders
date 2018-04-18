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
                 hidden_dim=512,
                 noise_dim=32,
                 ae_opt=Adam,
                 ae_learning_rate=0.0001,
                 critic_opt=Adam,
                 critic_learning_rate=0.0001,
                 mu=0.,
                 std_dev=1.0):
        # Define IO dimensions for models.
        self.img_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

        self.hidden_dim = hidden_dim
        self.ae_opt = ae_opt
        self.ae_learning_rate = ae_learning_rate
        self.critic_opt = critic_opt
        self.critic_learning_rate = critic_learning_rate

        # Latent space prior distribution variables.
        self.mu = mu
        self.sigma = std_dev

        # Component networks.
        self.encoder = self._construct_encoder()
        self.decoder = self._construct_decoder()
        self.generator = self._construct_generator()
        self.critic = self._construct_critic()

        self.autoencoder = self._construct_ae()
        self.gan = self._construct_gan()


    def _construct_encoder(self):
        """
        """
        img = Input(shape=self.img_dim)
        d1 = bn_conv_layer(img, self.img_dim[-1], 4, 2)
        d2 = bn_conv_layer(d1, 64, 4, 2)
        d3 = bn_conv_layer(d2, 128, 4, 2)
        d4 = bn_conv_layer(d3, 256, 4, 2)
        flat_1 = Flatten()(d4)
        fc_1 = bn_dense(flat_1, self.hidden_dim)
        z = Dense(self.latent_dim)(fc_1)

        encoder = Model(img, z)
        return encoder

    def _construct_decoder(self):
        """
        """
        z = Input(shape=(self.latent_dim,))
        z0 = Dense(self.hidden_dim)(z)
        z_reshp = Reshape((1, 1, self.hidden_dim))(z0)
        z1 = bn_deconv_layer(z_reshp, 256, 4, 4)
        z2 = bn_deconv_layer(z1, 128, 4, 2)
        z3 = bn_deconv_layer(z2, 64, 4, 2)
        z4 = bn_deconv_layer(z3, 32, 4, 2)
        gen_img = bn_deconv_layer(z4, self.img_dim[-1], 4, 2,
                                  activation='sigmoid',
                                  batchnorm=False)

        decoder = Model(z, gen_img)
        return decoder

    def _construct_generator(self):
        z = Input(shape=(self.noise_dim,))
        fc_3 = bn_dense(z, 64)
        fc_4 = bn_dense(fc_3, 128)
        fc_5 = bn_dense(fc_4, 256)
        code_fake = bn_dense(fc_5, self.latent_dim)

        generator = Model(z, code_fake)
        return generator

    def _construct_critic(self):
        code = Input(shape=(self.latent_dim,))
        fc_6 = bn_dense(code, 64, activation=None)
        lk_act_1 = LeakyReLU(0.2)(fc_6)
        fc_7 = bn_dense(lk_act_1, 32, activation=None)
        lk_act_2 = LeakyReLU(0.2)(fc_7)
        fc_8 = bn_dense(lk_act_2, 32, activation=None)
        lk_act_3 = LeakyReLU(0.2)(fc_8)
        real_prob = bn_dense(lk_act_3, 1, activation='sigmoid')

        critic = Model(code, real_prob)
        critic.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                       loss='binary_crossentropy')
        return critic

    def _construct_ae(self):
        """
        Construct AE and compile for training.
        """
        autoencoder = Model(self.encoder.input,
                            self.decoder(self.encoder.output))
        autoencoder.compile(optimizer=self.ae_opt(lr=self.ae_learning_rate),
                            loss='binary_crossentropy')
        return autoencoder

    def _construct_gan(self):
        """
        Construct GAN to teach generator to trick critic of latent space.
        """
        self.critic.trainable = False
        gan = Model(self.generator.input, self.critic(self.generator.output))
        gan.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
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
                x_pos, y_pos,
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
