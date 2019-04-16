"""
Implementation of Adversarial autoencoder.

Using ideas from https://arxiv.org/pdf/1511.05644.pdf
"""
from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD

import numpy as np

from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer)


class AAE(object):
    def __init__(self, input_dim, latent_dim,
                 hidden_dim=512,
                 reconst_weight=1.,
                 adv_weight=1.,
                 ae_opt=Adam,
                 ae_learning_rate=0.0001,
                 critic_opt=SGD,
                 critic_learning_rate=0.0001,
                 mu=0., std_dev=1.,
                 joint_train=False):
        # Save train configuration bool. It
        # controls if we traing the generator
        # and encoder with joint objectives
        # have separate graident updates b/w
        # reconstruction and generator trick loss.
        self.joint_train = joint_train

        # Define IO dimensions for models.
        self.img_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Misc variables.
        self.reconst_weight = reconst_weight
        self.adv_weight = adv_weight
        self.mu = mu
        self.std_dev = std_dev
        self.ae_opt = ae_opt
        self.ae_learning_rate = ae_learning_rate
        self.critic_opt = critic_opt
        self.critic_learning_rate = critic_learning_rate

        # Component networks.
        self.encoder = self._construct_encoder()
        self.decoder = self._construct_decoder()
        self.critic = self._construct_critic()

        self.autoencoder = self._construct_ae()
        self.gan = self._construct_gan()

    def _construct_encoder(self):
        """
        CNN encoder.
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
        CNN decoder.
        """
        z = Input(shape=(self.latent_dim,))
        z0 = Dense(self.hidden_dim)(z)
        z_reshp = Reshape((1, 1, self.hidden_dim))(z0)
        z1 = bn_deconv_layer(z_reshp, 256, 4, 4,
                             dropout_rate=0.2)
        z2 = bn_deconv_layer(z1, 128, 4, 2,
                             dropout_rate=0.2)
        z3 = bn_deconv_layer(z2, 64, 4, 2,
                             dropout_rate=0.2)
        z4 = bn_deconv_layer(z3, 32, 4, 2,
                             dropout_rate=0.2)
        gen_img = bn_deconv_layer(z4, self.img_dim[-1], 4, 2,
                                  activation='sigmoid',
                                  batchnorm=False)

        decoder = Model(z, gen_img)
        return decoder

    def _construct_critic(self):
        """
        FC Discriminator of latent.
        """
        z = Input(shape=(self.latent_dim,))
        fc_6 = bn_dense(z, 64, activation=None)
        lk_act_1 = LeakyReLU(0.2)(fc_6)
        fc_7 = bn_dense(lk_act_1, 32, activation=None)
        lk_act_2 = LeakyReLU(0.2)(fc_7)
        fc_8 = bn_dense(lk_act_2, 32, activation=None)
        lk_act_3 = LeakyReLU(0.2)(fc_8)
        real_prob = bn_dense(lk_act_3, 1, activation='sigmoid')

        critic = Model(z, real_prob)
        critic.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                       loss='binary_crossentropy')
        return critic

    def _construct_ae(self):
        """
        Construct AE and compile for training.

        We train the reconstruction and adversarial objectives together.
        """
        if self.joint_train:
            self.critic.trainable = False
            autoencoder = Model(self.encoder.input,
                                [self.decoder(self.encoder.output),
                                 self.critic(self.encoder.output)])
            autoencoder.compile(optimizer=self.ae_opt(lr=self.ae_learning_rate),
                                loss=['binary_crossentropy',
                                      'binary_crossentropy'],
                                loss_weights=[self.reconst_weight,
                                              self.adv_weight])
        else:
            autoencoder = Model(self.encoder.input,
                                self.decoder(self.encoder.output))
            autoencoder.compile(optimizer=self.ae_opt(lr=self.ae_learning_rate),
                                loss='mse')
        return autoencoder

    def _construct_gan(self):
        """
        Construct GAN to teach encoder/generator to trick critic.
        """
        self.critic.trainable = False
        gan = Model(self.encoder.input, self.critic(self.encoder.output))
        gan.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                    loss='binary_crossentropy')
        return gan

    def _prep_data(self, x_train):
        """
        Prep data and create vectors for training.
        """
        batch_size = x_train.shape[0]
        x_neg = self.encoder.predict_on_batch(x_train)
        y_neg = np.zeros(batch_size)
        x_pos = self.std_dev * np.random.randn(batch_size,
                                               self.latent_dim) + self.mu
        y_pos = np.ones(batch_size)
        if self.joint_train:
            y_train = [x_train, y_pos]
        else:
            y_train = x_train

        return (x_neg, y_neg, x_pos, y_pos, y_train)

    def train_on_batch(self, x_train):
        """
        One gradient training on input batch. 
        """
        # Prep data for batch update.
        (x_neg, y_neg,
         x_pos, y_pos,
         y_train) = self._prep_data(x_train)

        # 1. Train Autoencoder.
        loss = self.autoencoder.train_on_batch(x_train, y_train)
        if self.joint_train:
            _, ae_loss, gan_loss = loss
        else:
            ae_loss = loss

        # 2. Train Critic (Discriminator).
        # self.critic.trainable = True  # Unfreeze critic.
        critic_loss = self.critic.train_on_batch(x_neg, y_neg)
        critic_loss += self.critic.train_on_batch(x_pos, y_pos)

        # 3. Train Generator again on updated critic.
        # Note: this is not executed if autoencoder training
        # includes generator trick loss.
        if not self.joint_train:
            gan_loss = self.gan.train_on_batch(x_train, y_pos)

        sum_loss = ae_loss + critic_loss + gan_loss
        return (sum_loss, ae_loss, critic_loss, gan_loss)
