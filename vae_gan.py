"""
Implementation of VAE-GAN
"""
from keras import backend
from keras.layers import (Activation, Dense, Dropout)

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, SGD

import numpy as np

from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer)


class VAE_GAN(object):
    def __init__(self, input_dim, latent_dim,
                 ae_opt=Adam,
                 ae_learning_rate=0.0001,
                 critic_opt=Adam,
                 critic_learning_rate=0.0001):
        # Define IO dimensions for models.
        self.in_dim = input_dim
        self.latent_dim = latent_dim

        # Component networks.
        (self.encoder,
         self.vae_loss_fn) = self._construct_encoder(input_dim,
                                                     latent_dim)
        self.decoder = self._construct_decoder(latent_dim,
                                               input_dim)
        self.critic = self._construct_critic(input_dim,
                                             critic_opt,
                                             critic_learning_rate)

        self.gan = self._construct_gan(critic_opt, critic_learning_rate)
        self.autoencoder = self._construct_ae(ae_opt, ae_learning_rate)


    def _construct_encoder(self, in_dim, out_dim):
        x = Input(shape=in_dim)
        conv_1 = bn_conv_layer(x, 128, 3)
        conv_2 = bn_conv_layer(conv_1, 64, 3)
        conv_3 = bn_conv_layer(conv_2, 32, 3)
        flat_1 = Flatten()(conv_3)
        do = Dropout(0.3)(flat_1)
        fc_1 = bn_dense(do, 256)
        z_mu = Dense(out_dim)(fc_1)
        z_log_sigma = Dense(out_dim)(fc_1)

        def sample_z(args):
            z_mu, z_log_sigma = args
            epsilon = backend.random_normal(
                shape=backend.shape(z_mu),
                mean=0., stddev=1.0)
            return z_mu + backend.exp(z_log_sigma / 2) * epsilon

        def vae_loss_fn(x, x_decoded_mean):
            x = backend.flatten(x)
            x_decoded_mean = backend.flatten(x_decoded_mean)
            flat_dim = np.product(in_dim)
            xent_loss = flat_dim * binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * backend.sum(1 + z_log_sigma - backend.square(z_mu) - backend.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

        z = Lambda(sample_z)([z_mu, z_log_sigma])

        encoder = Model(x, z)
        return encoder, vae_loss_fn

    def _construct_decoder(self, in_dim, out_dim):
        z = Input(shape=(in_dim,))
        fc_2 = bn_dense(z, 256)

        shp = 8 * np.product(out_dim)
        ups_1 = Dense(shp, activation='relu')(fc_2)

        shp = (out_dim[0], out_dim[1], 8)
        reshp_1 = Reshape(shp)(ups_1)
        deconv_1 = bn_deconv_layer(reshp_1, 128, 3)
        deconv_2 = bn_deconv_layer(deconv_1, 64, 3)
        deconv_3 = bn_deconv_layer(deconv_2, 32, 3)
        x_reconst = Conv2DTranspose(out_dim[2], 3,
                                    padding='same',
                                    activation='sigmoid')(deconv_3)

        decoder = Model(z, x_reconst)
        return decoder

    def _construct_critic(self, in_dim,
                          critic_opt, critic_learning_rate):
        x_gen = Input(shape=in_dim)
        conv_1 = bn_conv_layer(x_gen, 64, 3, activation=None)
        lk_act_1 = LeakyReLU(0.2)(conv_1)
        conv_2 = bn_conv_layer(lk_act_1, 128, 3, activation=None)
        lk_act_2 = LeakyReLU(0.2)(conv_2)
        conv_3 = bn_conv_layer(lk_act_2, 512, 3, activation=None)
        lk_act_3 = LeakyReLU(0.2)(conv_3)
        flat = Flatten()(lk_act_3)
        real_prob = Dense(1, activation='sigmoid')(flat)

        critic = Model(x_gen, real_prob)
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
                            loss=self.vae_loss_fn)
        return autoencoder

    def _construct_gan(self, critic_opt, critic_learning_rate):
        """
        Construct GAN to teach decoder/generator to trick critic.
        """
        gan = Model(self.decoder.inputs[0], self.critic(self.decoder.output))
        gan.compile(optimizer=critic_opt(lr=critic_learning_rate),
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
        self.critic.trainable = False  # Freeze critic for gen training.
        ae_loss = self.autoencoder.train_on_batch(x_train, y_train)

        # 2. Train Critic (Discriminator).
        self.critic.trainable = True  # Unfreeze critic.
        critic_loss = self.critic.train_on_batch(x_neg, y_neg)
        critic_loss += self.critic.train_on_batch(x_pos, y_pos)

        # 3. Train Generator.
        self.critic.trainable = False
        gan_loss = self.gan.train_on_batch(
            self.encoder.predict_on_batch(x_train), y_pos)

        sum_loss = ae_loss + critic_loss + gan_loss
        return (sum_loss, ae_loss, critic_loss, gan_loss)
