"""
Implementation of WAE-MMD

See:
https://github.com/tolstikhin/wae/blob/master/wae.py
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

import numpy as np

from blocks import (bn_dense,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)


class WAE_MMD(object):
    def __init__(self, input_dim, latent_dim,
                 hidden_dim=512,
                 enc_param=64,
                 dec_param=16,
                 ae_opt=Adam,
                 ae_learning_rate=0.0002,
                 mu=0., std_dev=1.,
                 mmd_weight=1.):

        # Define IO dimensions for models.
        self.img_dim = input_dim
        self.latent_dim = latent_dim

        # Misc parameters.
        self.mmd_weight = mmd_weight
        self.hidden_dim = hidden_dim
        self.enc_param = enc_param
        self.dec_param = dec_param
        self.mu = mu
        self.std_dev = std_dev
        self.ae_opt = ae_opt
        self.ae_learning_rate = ae_learning_rate

        # Component networks.
        self.encoder, self.vae_loss_fn = self._construct_encoder()
        self.decoder = self._construct_decoder()
        self.autoencoder = self._construct_ae()


    def _construct_encoder(self):
        """
        CNN encoder with mmd loss.
        """
        img = Input(shape=self.img_dim)
        conv_block = convnet(img, self.enc_param)
        flat_1 = Flatten()(conv_block)
        fc_1 = bn_dense(flat_1, self.hidden_dim)
        z = Dense(self.latent_dim)(fc_1)

        def mmd_loss_fn(sample_qz, sample_pz):
            """
            Taken mmd loss implementation from
            https://github.com/tolstikhin/wae/blob/master/wae.py 
            """
            sigma2_p = 1. ** 2
            C_base = 2. * self.latent_dim * self.std_dev
            n = K.shape(sample_pz)[0]
            n = K.cast(n, 'int32')
            nf = K.cast(n, 'float32')

            norms_pz = K.sum(K.square(sample_pz), axis=1, keepdims=True)
            dotprods_pz = K.dot(sample_pz, K.transpose(sample_pz))
            distances_pz = norms_pz + K.transpose(norms_pz) - 2. * dotprods_pz

            norms_qz = K.sum(K.square(sample_qz), axis=1, keepdims=True)
            dotprods_qz = K.dot(sample_qz, K.transpose(sample_qz))
            distances_qz = norms_qz + K.transpose(norms_qz) - 2. * dotprods_qz

            dotprods = K.dot(sample_qz, K.transpose(sample_pz))
            distances = norms_qz + K.transpose(norms_pz) - 2. * dotprods

            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = C_base * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = K.sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = K.sum(res2) * 2. / (nf * nf)
                stat += res1 - res2

            return stat

        def vae_loss_fn(x, x_decoded):
            reconst_loss = binary_crossentropy(K.flatten(x),
                                               K.flatten(x_decoded))
            mmd_loss = mmd_loss_fn(z, K.random_normal(K.shape(z)))

            return reconst_loss + (self.mmd_weight * mmd_loss)

        encoder = Model(img, z)
        return encoder, vae_loss_fn

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
