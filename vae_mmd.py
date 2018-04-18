"""
Implementation of VAE-MMD

See:
http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html
"""
from keras import backend as K
from keras.layers import (Input, Activation,
                          Dense, Dropout,
                          Flatten, Lambda,
                          Reshape)
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

from blocks import (bn_dense,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)



def compute_kernel(x, y):
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x,
                               K.stack([x_size, 1, dim])),
                               K.stack([1, y_size, 1]))
    tiled_y = K.tile(K.reshape(y,
                               K.stack([1, y_size, dim])),
                               K.stack([x_size, 1, 1]))
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))


def mmd_loss_fn(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


class VAE_MMD(object):
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
        """
        img = Input(shape=self.img_dim)
        conv_block = convnet(img, self.enc_param)
        flat_1 = Flatten()(conv_block)
        fc_1 = bn_dense(flat_1, self.hidden_dim)
        z = Dense(self.latent_dim)(fc_1)

        def vae_loss_fn(x, x_decoded):
            reconst_loss = binary_crossentropy(K.flatten(x),
                                               K.flatten(x_decoded))
            mmd_loss = mmd_loss_fn(z, K.random_normal(K.shape(z)))

            return reconst_loss + (self.mmd_weight * mmd_loss)

        encoder = Model(img, z)
        return encoder, vae_loss_fn

    def _construct_decoder(self):
        """
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
