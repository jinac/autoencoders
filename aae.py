"""
Using ideas from https://arxiv.org/pdf/1511.05644.pdf
"""
from keras.layers import (Dense, Dropout, Flatten,
                          Input, Reshape)
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD

import numpy as np

from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer)


class AAE(object):
    def __init__(self, input_dim, latent_dim,
                 ae_opt=Adam,
                 ae_learning_rate=0.0001,
                 critic_opt=SGD,
                 critic_learning_rate=0.0001,
                 prior_mu=0.,
                 prior_sigma=1.0,
                 joint_train=False):
        # Save train configuration bool. It
        # controls if we traing the generator
        # and encoder with joint objectives
        # have separate graident updates b/w
        # reconstruction and generator trick loss.
        self.joint_train = joint_train

        # Define IO dimensions for models.
        self.in_dim = input_dim
        self.latent_dim = latent_dim

        # Component networks.
        self.encoder = self._construct_encoder(input_dim,
                                               latent_dim)
        self.decoder = self._construct_decoder(latent_dim,
                                               input_dim)
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
        z = Dense(out_dim)(fc_1)

        encoder = Model(x, z)
        return encoder

    def _construct_decoder(self, in_dim, out_dim):
        z = Input(shape=(in_dim,))
        fc_2 = bn_dense(z, 128)

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

        decoder = Model(z, x_reconst)
        return decoder

    def _construct_critic(self, in_dim,
                          critic_opt, critic_learning_rate):
        z = Input(shape=(in_dim,))
        fc_6 = bn_dense(z, 64, activation=None)
        lk_act_1 = LeakyReLU(0.2)(fc_6)
        fc_7 = bn_dense(lk_act_1, 32, activation=None)
        lk_act_2 = LeakyReLU(0.2)(fc_7)
        fc_8 = bn_dense(lk_act_2, 32, activation=None)
        lk_act_3 = LeakyReLU(0.2)(fc_8)
        real_prob = bn_dense(lk_act_3, 1, activation='sigmoid')

        critic = Model(z, real_prob)
        critic.compile(optimizer=critic_opt(lr=critic_learning_rate),
                       loss='binary_crossentropy')
        return critic

    def _construct_ae(self, ae_opt, ae_learning_rate):
        """
        Construct AE and compile for training.

        We train the reconstruction and adversarial objectives together.
        """
        if self.joint_train:
            autoencoder = Model(self.encoder.input,
                                [self.decoder(self.encoder.output),
                                 self.critic(self.encoder.output)])
            autoencoder.compile(optimizer=ae_opt(lr=ae_learning_rate),
                                loss=['binary_crossentropy',
                                      'binary_crossentropy'],
                                loss_weights=[1./3, 2./3])
        else:
            autoencoder = Model(self.encoder.input,
                                self.decoder(self.encoder.output))
            autoencoder.compile(optimizer=ae_opt(lr=ae_learning_rate),
                                loss='binary_crossentropy')
        return autoencoder

    def _construct_gan(self, critic_opt, critic_learning_rate):
        """
        Construct GAN to teach encoder/generator to trick critic.
        """
        gan = Model(self.encoder.input, self.critic(self.encoder.output))
        gan.compile(optimizer=critic_opt(lr=critic_learning_rate),
                    loss='binary_crossentropy')
        return gan

    def _prep_data(self, x_train):
        """
        Prep data and create vectors for training.
        """
        batch_size = x_train.shape[0]
        x_neg = self.encoder.predict_on_batch(x_train)
        y_neg = np.zeros(batch_size)
        x_pos = self.sigma * np.random.randn(batch_size,
                                             self.latent_dim) + self.mu
        y_pos = np.ones(batch_size)
        if self.joint_train:
            y_train = [x_train, y_pos]
        else:
            y_train = x_train

        return (x_neg, y_neg, x_pos, y_pos, y_train)

    def train_on_batch(self, x_train):
        # Prep data for batch update.
        (x_neg, y_neg,
         x_pos, y_pos,
         y_train) = self._prep_data(x_train)

        # 1. Train Autoencoder.
        self.critic.trainable = False  # Freeze critic for gen training.
        loss = self.autoencoder.train_on_batch(x_train, y_train)
        if self.joint_train:
            _, ae_loss, gan_loss = loss
        else:
            ae_loss = loss

        # 2. Train Critic (Discriminator).
        self.critic.trainable = True  # Unfreeze critic.
        critic_loss = self.critic.train_on_batch(x_neg, y_neg)
        critic_loss += self.critic.train_on_batch(x_pos, y_pos)

        # 3. Train Generator again on updated critic.
        # Note: this is not executed if autoencoder training
        # includes generator trick loss.
        if not self.joint_train:
            self.critic.trainable = False
            gan_loss = self.gan.train_on_batch(x_train, y_pos)

        sum_loss = ae_loss + critic_loss + gan_loss
        return (sum_loss, ae_loss, critic_loss, gan_loss)
