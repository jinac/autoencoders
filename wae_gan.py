"""
Implementation of WAE-GAN

See:
https://github.com/tolstikhin/wae/blob/master/wae.py
"""
from keras import backend
from keras.layers import (Input, Activation,
                          Dense, Dropout,
                          Flatten, Lambda,
                          Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

from loss import wasserstein_loss
from blocks import (bn_dense,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)


class WAE_GAN(object):
    def __init__(self, input_dim, latent_dim,
                 hidden_dim=512,
                 reconst_weight=1.,
                 critic_weight=1.,
                 enc_param=64,
                 dec_param=16,
                 ae_opt=Adam,
                 ae_learning_rate=0.0001,
                 critic_opt=SGD,
                 critic_learning_rate=0.0001,
                 mu=0., std_dev=1):
        # Define IO dimensions for models.
        self.img_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Misc variables.
        self.enc_param = enc_param
        self.dec_param = dec_param
        self.reconst_weight = reconst_weight
        self.critic_weight = critic_weight
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
        conv_block = convnet(img, self.enc_param)
        flat_1 = Flatten()(conv_block)
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
        deconv_block = deconvnet(z0, self.img_dim, self.dec_param)
        gen_img = bn_deconv_layer(deconv_block, self.img_dim[-1], 4, 2,
                                  activation='sigmoid', batchnorm=False)

        decoder = Model(z, gen_img)
        return decoder

    def _construct_critic(self):
        """
        FC Discriminator.
        """
        z = Input(shape=(self.latent_dim,))
        fc_6 = bn_dense(z, 128, activation='relu')
        fc_7 = bn_dense(fc_6, 64, activation='relu')
        fc_8 = bn_dense(fc_7, 64, activation='relu')
        real_prob = bn_dense(fc_8, 1, activation='linear')

        critic = Model(z, real_prob)
        critic.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                       loss=[wasserstein_loss], loss_weights=[self.critic_weight])
        return critic

    def _construct_ae(self):
        """
        Construct AE and compile for training.

        We train the reconstruction and adversarial objectives together.
        """
        autoencoder = Model(self.encoder.input,
                            self.decoder(self.encoder.output))
        autoencoder.compile(optimizer=self.ae_opt(lr=self.ae_learning_rate),
                            loss='binary_crossentropy')
        return autoencoder

    def _construct_gan(self):
        """
        Construct GAN to teach encoder/generator to trick critic.
        """
        self.critic.trainable = False
        gan = Model(self.encoder.input,
                    [self.decoder(self.encoder.output),
                     self.critic(self.encoder.output)])
        gan.compile(optimizer=self.critic_opt(lr=self.critic_learning_rate),
                    loss=['binary_crossentropy', wasserstein_loss],
                    loss_weights=[self.reconst_weight, self.critic_weight])
        return gan

    def _prep_data(self, x_train):
        """
        Prep data and create vectors for training.
        """
        batch_size = x_train.shape[0]
        x_neg = self.encoder.predict_on_batch(x_train)
        y_neg = np.ones(batch_size)
        x_pos = np.random.normal(self.mu, self.std_dev,
                                 [batch_size, self.latent_dim])
        y_pos = -np.ones(batch_size)
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

        # 1. Train Critic (Discriminator).
        # self.critic.trainable = True  # Unfreeze critic.
        critic_loss = self.critic.train_on_batch(x_neg, y_neg)
        critic_loss += self.critic.train_on_batch(x_pos, y_pos)

        # 2. Train autoencoder on updated critic.
        # Note: this is not executed if autoencoder training
        # includes generator trick loss.
        gan_loss, ae_loss, reg_loss = self.gan.train_on_batch(x_train, [x_train, y_neg])

        sum_loss = ae_loss + critic_loss + reg_loss
        return (sum_loss, ae_loss, critic_loss, reg_loss)
