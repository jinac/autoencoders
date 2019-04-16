"""
Implementation of Beta VAE
"""
from keras import backend as K
from keras.layers import Dense, Flatten, Input, Lambda
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

from blocks import (bn_dense,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)


class B_VAE(object):
    def __init__(self, input_dim, latent_dim,
                 hidden_dim=512,
                 beta=1,
                 enc_param=64,
                 dec_param=16,
                 ae_opt=Adam,
                 ae_learning_rate=0.0002,
                 mu=0., std_dev=1.):

        # Define IO dimensions for models.
        self.img_dim = input_dim
        self.latent_dim = latent_dim

        # Misc parameters.
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.enc_param = enc_param
        self.dec_param = dec_param
        self.mu = mu
        self.std_dev = std_dev
        self.ae_opt = ae_opt
        self.ae_learning_rate = ae_learning_rate

        # Component networks.
        (self.encoder,
         self.vae_loss_fn) = self._construct_encoder()
        self.decoder = self._construct_decoder()
        self.autoencoder = self._construct_ae()


    def _construct_encoder(self):
        """
        CNN encoder.
        """
        img = Input(shape=self.img_dim)
        conv_block = convnet(img, self.enc_param, bias=False)
        flat_1 = Flatten()(conv_block)
        fc_1 = bn_dense(flat_1, self.hidden_dim)
        z_mu = Dense(self.latent_dim)(fc_1)
        z_log_sigma = Dense(self.latent_dim)(fc_1)

        def sample_z(args):
            mu, log_sigma = args
            epsilon = K.random_normal(shape=K.shape(mu),
                                      mean=self.mu,
                                      stddev=self.std_dev)
            return mu + K.exp(log_sigma / 2) * epsilon

        def vae_loss_fn(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            flat_dim = np.product(self.img_dim)
            reconst_loss = flat_dim * binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * K.sum(
                1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
            return reconst_loss + (self.beta * kl_loss)

        z = Lambda(sample_z)([z_mu, z_log_sigma])

        encoder = Model(img, z)
        return encoder, vae_loss_fn

    def _construct_decoder(self):
        """
        CNN decoder.
        """
        z = Input(shape=(self.latent_dim,))
        z0 = Dense(self.hidden_dim)(z)
        deconv_block = deconvnet(z0, self.img_dim, self.dec_param, bias=False)
        gen_img = bn_deconv_layer(deconv_block, self.img_dim[-1], 4, 2,
                                  activation='sigmoid', batchnorm=False)

        decoder = Model(z, gen_img)
        return decoder

    def _construct_ae(self):
        """
        Construct AE and compile for training.
        """
        autoencoder = Model(self.encoder.input,
                            self.decoder(self.encoder.output))
        autoencoder.compile(optimizer=self.ae_opt(lr=self.ae_learning_rate),
                            loss=self.vae_loss_fn)
        return autoencoder


    def train_on_batch(self, x_train):
        """
        One gradient training on input batch. 
        """
        return self.autoencoder.train_on_batch(x_train, x_train)
